"""Iterative legal-document search agent backed by Elasticsearch and Milvus."""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from core import settings
from tools import (
    search_law_terms,
    search_law_terms_in_document,
    search_legal_documents,
)


def _load_legal_statuses() -> list[str]:
    path = Path(__file__).parents[1] / "data" / "tinh_trang_hieu_luc.json"
    with path.open(encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise RuntimeError(f"Danh mục tình trạng hiệu lực không hợp lệ: {path}")
    return values


LEGAL_STATUSES = _load_legal_statuses()
CURRENT_RULE_STATUSES = ["Còn hiệu lực", "Chưa có hiệu lực", "Hết hiệu lực một phần"]
LEGAL_STATUSES_TEXT = json.dumps(LEGAL_STATUSES, ensure_ascii=False)
CURRENT_RULE_STATUSES_TEXT = json.dumps(CURRENT_RULE_STATUSES, ensure_ascii=False)


class LegalSearchState(MessagesState, total=False):
    search_loops: int


def _with_description(tool: BaseTool, description: str) -> BaseTool:
    return tool.model_copy(update={"description": description})


tools = [
    _with_description(
        search_legal_documents,
        """Tìm candidate văn bản pháp luật bằng BM25 trên tiêu đề và toàn văn.
Dùng cho câu hỏi về hành vi, chế tài, điều kiện hoặc cụm từ pháp lý cụ thể. Viết lại query ngắn,
sửa lỗi chính tả và chuyển cách nói đời thường thành thuật ngữ pháp lý. Có thể dùng các filter ngày,
tình trạng hiệu lực, số hiệu và don_vi khi người dùng cung cấp hoặc ngữ cảnh yêu cầu. Chỉ dùng
don_vi="Trung ương" khi cần giới hạn ở văn bản cấp trung ương. Không dùng don_vi để lọc văn bản địa
phương vì giá trị chưa được chuẩn hóa và có nhiều cách biểu diễn. Kết quả trả ID văn bản, score,
metadata và highlight; ID này có thể được dùng ở bước kiểm chứng sau.""",
    ),
    _with_description(
        search_law_terms,
        """Tìm candidate trên toàn bộ điều, khoản và thuật ngữ pháp luật trong Milvus.
Tool này chỉ dành cho bước tìm kiếm toàn cục và không nhận doc_id. Có thể lọc theo tình trạng hiệu lực,
cơ quan ban hành, loại văn bản và số hiệu. Viết query thành đối tượng + hành vi pháp lý chuẩn hóa +
nhu cầu tra cứu; ví dụ đổi "vượt đèn đỏ" thành "người điều khiển xe ô tô không chấp hành hiệu lệnh
của đèn tín hiệu giao thông". Kết quả trả doc_id của các văn bản ứng viên cùng nội dung điều khoản.""",
    ),
    _with_description(
        search_law_terms_in_document,
        """Kiểm chứng một candidate bằng cách tìm điều, khoản bên trong đúng văn bản đó.
Chỉ gọi ở vòng sau khi doc_id đã xuất hiện trong kết quả search_legal_documents hoặc
search_law_terms của vòng trước. doc_id là bắt buộc và không được tự đoán. Nếu kết quả chỉ là các điều
chung hoặc không chứa đúng đối tượng/hành vi, candidate không đạt và phải quay lại tìm candidate khác.""",
    ),
]


instructions = f"""Bạn là agent truy hồi văn bản pháp luật Việt Nam. Ngày hiện tại là
{datetime.now().strftime("%Y-%m-%d")}.

Nhiệm vụ duy nhất là xác định một hoặc nhiều văn bản đích phù hợp; không trả lời nội dung pháp luật
trong câu hỏi. Chỉ được kết luận dựa trên kết quả tool và luôn search ít nhất một lần.

QUY TRÌNH TOOL BẮT BUỘC:
1. Không sử dụng search_law_titles vì kết quả thiếu context và dễ nhiễu.
2. Bước tìm candidate chỉ dùng search_legal_documents hoặc search_law_terms. Ở bước này chỉ dùng
   filter nghiệp vụ như ngày, tình trạng hiệu lực, cơ quan ban hành, loại văn bản, số hiệu và đơn vị.
   Mỗi tool call phải có search_reason giải thích ngắn intent đang tìm, vì sao chọn tool đó, và vì sao
   dùng hoặc không dùng từng filter. Đây là kế hoạch search trước khi có kết quả, không được khẳng định
   dữ kiện chưa được tool xác nhận.
3. search_law_terms là tìm kiếm toàn cục và không có filter doc_id.
4. Chỉ ở vòng sau, khi đã có candidate ID từ kết quả tool, mới gọi search_law_terms_in_document với
   đúng doc_id đó để tìm điều trong văn bản. Không gọi tìm candidate và kiểm chứng trong cùng một vòng.
   Không tự tạo hoặc suy đoán doc_id.
5. Candidate chỉ được xác nhận khi kết quả trong văn bản chứa đúng đối tượng và hành vi. Các điều chung
   về đối tượng, thủ tục, mức phạt hoặc nguyên tắc xử phạt không đủ để xác nhận.

QUY TẮC SEARCH VÀ ĐÁNH GIÁ:
- Trước khi search, tách câu hỏi có nhiều đối tượng, hành vi hoặc thủ tục thành các intent độc lập
  cần được bao phủ. Có thể gọi nhiều tool tìm candidate trong cùng một vòng, mỗi tool call cho một
  intent; không gộp các intent thành query làm mất một vế.
- Phải có kết quả tool phù hợp cho từng intent trước khi dừng. Một văn bản chỉ được xem là bao phủ
  nhiều intent khi bằng chứng từ tool thể hiện riêng từng intent; không được suy ra intent chưa search
  từ việc văn bản đã khớp intent khác.
- Ví dụ, "đăng ký tạm trú tạm vắng" phải được chuẩn hóa và tìm riêng thành "hồ sơ thủ tục đăng ký
  tạm trú" và "trình tự thủ tục khai báo tạm vắng". Không dừng chỉ vì đã tìm thấy một trong hai.
- Nếu còn intent chưa được bao phủ thì phải viết lại query và search tiếp. Nếu hết vòng mà vẫn thiếu
  bất kỳ intent bắt buộc nào, trả documents rỗng thay vì trả kết quả mới chỉ bao phủ một phần.
- Chỉ coi là câu hỏi về quy tắc hiện hành khi câu hỏi xác định rõ việc áp dụng pháp luật cho một tình
  huống, hoặc hỏi rõ quy định đang/chuẩn bị áp dụng, ví dụ: "bị phạt bao nhiêu", "hiện nay có được
  phép không", "điều kiện đang áp dụng". Khi đó PHẢI truyền chính xác và đầy đủ
  tinh_trang_hieu_luc={CURRENT_RULE_STATUSES_TEXT}; không được rút gọn còn một hoặc hai giá trị. Tool
  sẽ kết hợp ba giá trị theo OR. Ví dụ "ô tô vượt đèn đỏ hiện nay bị phạt bao nhiêu" phải dùng đúng
  danh sách ba giá trị này.
- Ngoài trường hợp trên, khuyến khích search tự do và để tinh_trang_hieu_luc=null nhằm giữ độ phủ.
  Tra cứu lịch, sự kiện, ngày nghỉ, chủ đề, tiêu đề, số hiệu, cơ quan hoặc một năm cụ thể không tự nó
  phải là câu hỏi quy tắc hiện hành, và không được mặc định đồng nhất "mới" với "Còn hiệu lực".
- Nếu người dùng nêu rõ tình trạng cần tìm thì dùng đúng tình trạng đó, kể cả khi nằm ngoài nhóm hiện
  hành; trường hợp này có thể truyền danh sách một phần tử. Các giá trị tình trạng hiệu lực hiện có
  trong dữ liệu là: {LEGAL_STATUSES_TEXT}.
- Filter tình trạng là filter chính xác và có thể làm mất candidate. Nếu kết quả đã lọc không đáp ứng
  ràng buộc chính, phải thử lại không có filter tình trạng khi vẫn còn vòng search.
- Không thêm số hiệu văn bản nếu số hiệu đó chưa có trong câu hỏi hoặc kết quả tool.
- Với search_legal_documents, dùng don_vi="Trung ương" khi người dùng chỉ cần văn bản cấp trung ương
  như luật, nghị định, hiến pháp, thông tư. Không dùng don_vi cho yêu cầu địa phương vì dữ liệu chưa được chuẩn
  hóa; đưa địa danh hoặc cơ quan vào query nếu cần. Không tự thêm don_vi nếu câu hỏi không giới hạn
  phạm vi ban hành.
- Chuẩn hóa cách nói đời thường thành thuật ngữ pháp lý; không giữ từ đa nghĩa đứng riêng. Ví dụ đổi
  "vượt đèn đỏ" thành "không chấp hành hiệu lệnh của đèn tín hiệu giao thông".
- Sau mỗi vòng, kiểm tra kết quả có thật sự khớp đối tượng, hành vi và nhu cầu hay chỉ khớp từ chung.
- Trước khi kết luận, đối chiếu lại candidate với toàn bộ ràng buộc trong câu hỏi, đặc biệt là loại
  văn bản và tình trạng hiệu lực. 
- Nếu chưa chắc chắn, đổi query hoặc tool; không lặp nguyên query với cùng tool sau kết quả sai.
- Nếu chưa tìm được candidate mong muốn, ngoài việc đổi tool hoặc cách diễn đạt query, phải cân nhắc
  nới rộng truy vấn để tăng recall: rút gọn query về đối tượng/hành vi cốt lõi, dùng thuật ngữ pháp lý
  tổng quát hơn, hoặc bỏ bớt filter không do người dùng yêu cầu và không phải ràng buộc bắt buộc.
  Không được nới rộng bằng cách bỏ qua đối tượng, hành vi, phạm vi hoặc điều kiện mà người dùng đã nêu
  rõ. Mỗi lần nới rộng phải giải thích ngắn trong search_reason.
- Không xem top-k hoặc score cao là bằng chứng duy nhất. Nếu kiểm chứng thất bại, loại candidate và
  quay lại search candidate khác nếu còn vòng.

ĐẦU RA:
- Không trả lời mức phạt, quyền, nghĩa vụ, thời hạn, điều kiện hoặc nội dung tư vấn.
- Khi đủ bằng chứng, chỉ trả đúng một JSON object, không markdown. Trường reasoning phải là kết luận
  kiểm chứng ngắn: nêu các ràng buộc chính từ câu hỏi, vì sao văn bản được chọn đáp ứng chúng và, khi
  cần, vì sao candidate dễ nhầm không phải văn bản đích. Không trình bày chuỗi suy luận chi tiết:
  {{"documents": [{{"id": "ID", "title": "Tên văn bản", "so_hieu": "Số hiệu hoặc null",
  "tinh_trang_hieu_luc": "Tình trạng hoặc null"}}],
  "reasoning": "Kết luận kiểm chứng ngắn dựa trên kết quả tool"}}.
- Chỉ điền dữ liệu xuất hiện trong kết quả tool; trường không có dữ liệu phải là null.
- Nếu cần nhiều văn bản, trả nhiều phần tử đã khử trùng lặp.
- Nếu hết vòng mà chưa có candidate đủ tin cậy, trả {{"documents": [], "reasoning": "Lý do ngắn
  cho biết bằng chứng nào còn thiếu hoặc ràng buộc nào chưa được đáp ứng"}}.
"""


@lru_cache(maxsize=1)
def get_legal_search_model() -> BaseChatModel:
    if settings.GEMINI_API is None:
        raise RuntimeError("Thiếu cấu hình GEMINI_API cho legal-search-agent.")
    if not settings.GEMINI_MODEL:
        raise RuntimeError("Thiếu cấu hình GEMINI_MODEL cho legal-search-agent.")
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        api_key=settings.GEMINI_API,
        temperature=0.1,
        streaming=True,
    )


def normalize_final_response(response: AIMessage) -> AIMessage:
    if response.tool_calls or isinstance(response.content, str):
        return response
    text = "".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return response.model_copy(update={"content": text})


async def call_model(state: LegalSearchState, config: RunnableConfig) -> LegalSearchState:
    search_loops = state.get("search_loops", 0)
    max_loops = settings.LEGAL_SEARCH_MAX_LOOPS
    messages = [
        SystemMessage(content=instructions),
        SystemMessage(
            content=f"Đã dùng {search_loops}/{max_loops} vòng search."
            + (
                " Đã đạt giới hạn: không gọi tool nữa; kết luận chỉ từ bằng chứng hiện có."
                if search_loops >= max_loops
                else ""
            )
        ),
        *state["messages"],
    ]
    model = get_legal_search_model()
    runnable = model if search_loops >= max_loops else model.bind_tools(tools)
    response = await runnable.ainvoke(messages, config)
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(response)}")
    return {"messages": [normalize_final_response(response)]}


tool_node = ToolNode(tools)


def _tool_message_payload(message: ToolMessage) -> Any:
    content = message.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def known_candidate_ids(state: LegalSearchState) -> set[str]:
    candidate_ids: set[str] = set()
    for message in state["messages"][:-1]:
        if not isinstance(message, ToolMessage):
            continue
        if message.name not in {"search_legal_documents", "search_law_terms"}:
            continue
        payload = _tool_message_payload(message)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            doc_info = item.get("doc_info") or {}
            document_id = doc_info.get("id") or item.get("doc_id")
            if document_id is not None:
                candidate_ids.add(str(document_id))
    return candidate_ids


async def call_tools(state: LegalSearchState, config: RunnableConfig) -> dict[str, Any]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")

    candidates = known_candidate_ids(state)
    valid_calls = []
    rejected_calls = []
    for call in last_message.tool_calls:
        if call["name"] != "search_law_terms_in_document":
            valid_calls.append(call)
        elif str(call.get("args", {}).get("doc_id")) in candidates:
            valid_calls.append(call)
        else:
            rejected_calls.append(call)

    tool_messages: list[Any] = []
    if valid_calls:
        validated_message = last_message.model_copy(update={"tool_calls": valid_calls})
        validated_state = {
            **state,
            "messages": [*state["messages"][:-1], validated_message],
        }
        result = await tool_node.ainvoke(validated_state, config)
        if not isinstance(result, dict):
            raise TypeError(f"Expected ToolNode dict output, got {type(result)}")
        tool_messages.extend(result.get("messages", []))

    for call in rejected_calls:
        tool_messages.append(
            ToolMessage(
                content=(
                    "Từ chối kiểm chứng: doc_id phải xuất hiện trong kết quả search candidate của "
                    "vòng trước. Hãy gọi search_legal_documents hoặc search_law_terms trước."
                ),
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )
        )

    return {
        "messages": tool_messages,
        "search_loops": state.get("search_loops", 0) + 1,
    }


def route_after_model(state: LegalSearchState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    return "tools" if last_message.tool_calls else "done"


builder = StateGraph(LegalSearchState)
builder.add_node("model", call_model)
builder.add_node("tools", call_tools)
builder.set_entry_point("model")
builder.add_edge("tools", "model")
builder.add_conditional_edges("model", route_after_model, {"tools": "tools", "done": END})

legal_search_agent = builder.compile()

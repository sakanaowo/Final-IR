"""
Mock Teacher Server - Giả lập server giảng viên để test RAG pipeline.
Usage: uv run python mock_teacher.py

Flow:
  1. Student gọi POST /api/v1/competition/register -> đăng ký server
  2. Student gọi POST /api/v1/competition/evaluate -> bắt đầu thi
     - Teacher gửi POST /upload tới student nếu document_received=false (timeout 120s)
     - Teacher gửi POST /ask 10 lần (mỗi câu timeout 60s)
  3. Student gọi POST /api/v1/competition/reset -> reset
  4. Student gọi GET /api/v1/competition/result -> xem kết quả
"""

import os
import time
import json
import logging
import threading

import requests
import uvicorn
from fastapi import FastAPI, Header, Request
from pydantic import BaseModel
from typing import Optional

from dotenv import load_dotenv

load_dotenv(".env.local")

TEACHER_HOST = "0.0.0.0"
TEACHER_PORT = int(os.environ.get("TEACHER_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mock_teacher")

app = FastAPI(title="Mock Teacher Server")

# ============================================================
# DOCUMENT - Trích đoạn dài từ PDF data (tiếng Việt + tiếng Anh)
# ============================================================
DOCUMENT_TEXT = """
ReAct Agent và Phương pháp Reasoning kết hợp Acting trong các Mô hình Ngôn ngữ Lớn

Phần 1: Giới thiệu về AI Agents

Mô hình ngôn ngữ lớn (LLM) đã tạo ra một bước đột phá trong khả năng xử lý ngôn ngữ tự nhiên. Tuy nhiên, để ứng dụng hiệu quả vào các tác vụ thực tế đòi hỏi khả năng tư duy nhiều bước, hành động có mục tiêu và phản ứng linh hoạt, chúng ta cần triển khai AI agents – những cấu trúc có khả năng lập kế hoạch, hành động, quan sát và suy luận. Hiện nay có nhiều thiết kế cho phép các mô hình này hoạt động theo các cách tiếp cận khác nhau:

- ReAct: Reasoning and Acting – kết hợp suy nghĩ và hành động qua nhiều vòng lặp.
- Plan-and-Execute: tách bạch hoàn toàn giữa giai đoạn lên kế hoạch và thực thi.

Phần 2: ReAct Agent

ReAct (Reasoning + Acting) là một phương pháp kết hợp giữa suy luận và hành động, cho phép các mô hình ngôn ngữ lớn (LLMs) vừa lý luận về các nhiệm vụ vừa thực hiện hành động liên quan đến nhiệm vụ đó. ReAct lần đầu tiên được giới thiệu trong bài báo: "ReAct: Synergizing Reasoning and Acting in Language Models".

Các mô hình chỉ có reasoning (suy luận) thường dựa hoàn toàn vào khả năng ngôn ngữ của LLM để tạo ra lời giải. Mặc dù chúng có thể mô phỏng tư duy logic, nhưng không thể tương tác với thế giới bên ngoài, như tìm kiếm thông tin mới, gọi công cụ, hay xác minh giả định. Điều này khiến chúng dễ bị giới hạn trong những gì đã được học trong quá trình huấn luyện, và dễ sinh ra câu trả lời "có vẻ hợp lý nhưng sai".

Ngược lại, các mô hình chỉ có acting (hành động) – ví dụ như các hệ thống gán tool cho prompt một cách trực tiếp – có thể thực thi tác vụ nhanh chóng, nhưng lại thiếu năng lực suy luận để quyết định khi nào nên gọi tool nào, hoặc nên dừng lại khi đã đủ thông tin.

ReAct kết hợp cả hai thế giới: Nó cho phép mô hình "nghĩ thành tiếng" trước khi ra quyết định hành động, và sau khi hành động xong, quan sát kết quả để điều chỉnh suy luận tiếp theo. Điều này tạo ra một vòng lặp Thought → Action → Observation, giúp mô hình:
- Hiểu rõ về bối cảnh trước khi hành động
- Tự sửa sai nếu kết quả quan sát không như mong đợi
- Chọn lựa hành động một cách có lý do, thay vì theo rule cố định
- Kết hợp nhiều công cụ linh hoạt, ví dụ: tra cứu → tính toán → lập kế hoạch → trả lời

Phương pháp ReAct hoạt động dựa trên việc kết hợp:
- Reasoning (Suy luận): Mô hình suy nghĩ về vấn đề, phân tích tình huống và lập kế hoạch cho hành động tiếp theo.
- Acting (Hành động): Mô hình thực hiện các hành động dựa trên suy luận của mình, thu thập thông tin mới và tiếp tục quá trình.

ReAct thực hiện một quy trình lặp đi lặp lại gồm các bước sau:
1. Suy luận (Thought): Mô hình suy nghĩ về tình huống hiện tại, phân tích các thông tin đã có và cân nhắc các bước tiếp theo.
2. Hành động (Action): Dựa trên suy luận, mô hình thực hiện một hành động cụ thể (ví dụ: tìm kiếm thông tin, gọi API, thực hiện tính toán).
3. Quan sát (Observation): Sau khi thực hiện hành động, mô hình quan sát kết quả thu được.
4. Tiếp tục chu trình: Quay lại bước suy luận với thông tin và quan sát mới, tiếp tục quá trình cho đến khi đạt được mục tiêu.

Phần 3: Plan-and-Execute Agent

Plan-and-Execute là một kiến trúc agent khác, tách bạch hoàn toàn giữa hai giai đoạn:
- Giai đoạn lập kế hoạch (Planning): Agent sử dụng LLM để phân tích yêu cầu và tạo ra một kế hoạch gồm nhiều bước tuần tự.
- Giai đoạn thực thi (Execution): Agent thực hiện từng bước trong kế hoạch, sử dụng các công cụ và thu thập kết quả.
- Giai đoạn đánh giá lại (Replanning): Sau mỗi bước thực thi, agent có thể đánh giá lại kế hoạch và điều chỉnh nếu cần.

Sự khác biệt chính giữa ReAct và Plan-and-Execute:
- ReAct: Suy luận và hành động xen kẽ trong từng bước, linh hoạt nhưng có thể thiếu tầm nhìn tổng thể.
- Plan-and-Execute: Lập kế hoạch tổng thể trước, sau đó thực thi tuần tự, có tầm nhìn xa hơn nhưng kém linh hoạt hơn khi gặp tình huống bất ngờ.

Phần 4: Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) là một kỹ thuật kết hợp giữa truy xuất thông tin (retrieval) và sinh văn bản (generation) để cải thiện độ chính xác của các mô hình ngôn ngữ lớn. Thay vì dựa hoàn toàn vào kiến thức nội tại đã được huấn luyện, RAG cho phép mô hình truy cập vào nguồn tri thức bên ngoài (knowledge base) tại thời điểm suy luận.

Quy trình RAG cơ bản:
1. Indexing: Tài liệu được chia thành các đoạn nhỏ (chunks), mỗi đoạn được chuyển thành vector embedding và lưu vào vector database.
2. Retrieval: Khi nhận được câu hỏi, hệ thống chuyển câu hỏi thành vector embedding và tìm kiếm các đoạn tài liệu có nội dung tương tự nhất trong vector database.
3. Augmented Generation: Các đoạn tài liệu liên quan được đưa vào prompt cùng với câu hỏi, và LLM sử dụng thông tin này để sinh câu trả lời chính xác hơn.

Các thành phần chính của hệ thống RAG:
- Embedding Model: Mô hình chuyển đổi văn bản thành vector số (ví dụ: sentence-transformers, vietnamese-sbert). Embedding model đóng vai trò quan trọng trong việc biểu diễn ngữ nghĩa của văn bản.
- Vector Database: Cơ sở dữ liệu lưu trữ và tìm kiếm vector (ví dụ: ChromaDB, FAISS, Pinecone, Qdrant). Vector database cho phép tìm kiếm nhanh các đoạn văn bản có ngữ nghĩa tương tự.
- Chunking Strategy: Chiến lược chia tài liệu thành các đoạn nhỏ. Có nhiều phương pháp chunking: theo kích thước cố định, theo câu, theo đoạn văn, hoặc theo ngữ nghĩa (semantic chunking).
- Reranker: Mô hình cross-encoder dùng để sắp xếp lại kết quả retrieval, giúp chọn ra các đoạn văn bản liên quan nhất. Reranker thường chính xác hơn bi-encoder nhưng chậm hơn.
- LLM: Mô hình ngôn ngữ lớn dùng để sinh câu trả lời cuối cùng dựa trên context đã truy xuất.

Phần 5: Knowledge Graph và GraphRAG

Knowledge Graph (Đồ thị tri thức) là một biểu diễn có cấu trúc của các thực thể (entities), thuộc tính (attributes) và mối quan hệ (relationships) giữa chúng. Knowledge graph cung cấp một framework ngữ nghĩa kết nối dữ liệu có cấu trúc và phi cấu trúc.

GraphRAG là sự kết hợp giữa Knowledge Graph và RAG, cho phép:
- Truy xuất thông tin chính xác hơn thông qua các mối quan hệ trong đồ thị
- Cung cấp context phong phú hơn cho LLM
- Hỗ trợ các câu hỏi phức tạp đòi hỏi nhiều bước suy luận
- Giảm hallucination bằng cách grounding câu trả lời vào dữ liệu thực tế

Trong hệ thống GraphRAG, dữ liệu được tổ chức thành:
- Nodes: Đại diện cho các thực thể (người, tổ chức, khái niệm, sự kiện)
- Edges: Đại diện cho các mối quan hệ giữa các thực thể
- Properties: Thuộc tính bổ sung cho nodes và edges

Microsoft đã phát triển một implementation cụ thể của GraphRAG sử dụng community detection algorithms để tạo các community summaries, cho phép trả lời các câu hỏi tổng hợp (global questions) hiệu quả hơn so với RAG truyền thống.

Phần 6: Xử lý Ngôn ngữ Tự nhiên (NLP)

NLP Pipeline bao gồm các bước chính:
1. Sentence Segmentation: Tách đoạn văn thành các câu riêng biệt.
2. Tokenization: Tách câu thành các token (từ, sub-word hoặc ký tự).
3. Part-of-Speech Tagging: Gán nhãn từ loại cho mỗi token (danh từ, động từ, tính từ...).
4. Named Entity Recognition (NER): Nhận dạng và phân loại các thực thể có tên (người, tổ chức, địa điểm, ngày tháng...).
5. Dependency Parsing: Phân tích cú pháp phụ thuộc, xác định mối quan hệ ngữ pháp giữa các từ trong câu.
6. Coreference Resolution: Xác định các đại từ hoặc cụm từ nào tham chiếu đến cùng một thực thể.

Trong tiếng Việt, NLP có những thách thức riêng biệt:
- Tách từ (Word Segmentation): Tiếng Việt không có dấu cách giữa các từ phức, ví dụ "học sinh" là một từ gồm hai tiếng.
- Dấu thanh (Tone marks): Tiếng Việt có 6 thanh điệu ảnh hưởng đến ý nghĩa từ.
- Từ đồng âm khác nghĩa: Nhiều từ có cách viết giống nhau nhưng nghĩa khác nhau tùy ngữ cảnh.

Phần 7: Vector Similarity Search

Tìm kiếm tương tự vector là phương pháp cốt lõi trong RAG. Các phương pháp phổ biến:
- Cosine Similarity: Đo độ tương tự dựa trên góc giữa hai vector. Giá trị từ -1 đến 1, với 1 là hoàn toàn giống nhau.
- Euclidean Distance: Đo khoảng cách Euclidean giữa hai vector. Giá trị càng nhỏ càng giống nhau.
- Dot Product: Tích vô hướng giữa hai vector. Thường dùng khi vector đã được chuẩn hóa.

Hybrid Search kết hợp nhiều phương pháp tìm kiếm:
- Sparse retrieval (BM25): Tìm kiếm dựa trên từ khóa, hiệu quả với các truy vấn chứa thuật ngữ chuyên ngành.
- Dense retrieval (Vector Search): Tìm kiếm dựa trên ngữ nghĩa, hiệu quả với các truy vấn mang tính diễn giải.
- Reciprocal Rank Fusion (RRF): Phương pháp kết hợp kết quả từ nhiều hệ thống retrieval khác nhau.

Phần 8: Advanced Retrieval Strategies

Các chiến lược retrieval nâng cao bao gồm:
- Multi-query retrieval: Sinh nhiều biến thể câu hỏi để truy vấn, tăng coverage.
- Contextual compression: Nén context để chỉ giữ lại thông tin liên quan nhất.
- Parent-child chunking: Truy xuất chunk con nhưng cung cấp chunk cha lớn hơn làm context.
- Hypothetical Document Embedding (HyDE): Sinh document giả từ câu hỏi, dùng embedding của document giả để truy vấn.
- Sentence Window Retrieval: Truy xuất câu cụ thể nhưng mở rộng window xung quanh để lấy thêm context.

Phần 9: Agentic RAG

Agentic RAG kết hợp khả năng của AI agents với RAG pipeline:
- Agent có thể quyết định khi nào cần truy vấn knowledge base
- Agent có thể reformulate câu hỏi nếu kết quả retrieval không đủ tốt
- Agent có thể sử dụng nhiều nguồn dữ liệu khác nhau
- Agent có thể thực hiện multi-hop reasoning qua nhiều bước truy vấn

Phần 10: RAG Application Evaluation

Đánh giá hệ thống RAG bao gồm nhiều khía cạnh:
- Retrieval Quality: Đánh giá chất lượng các đoạn văn bản được truy xuất (Precision@K, Recall@K, MRR).
- Answer Quality: Đánh giá chất lượng câu trả lời cuối cùng (Faithfulness, Answer Relevancy, Correctness).
- Context Relevance: Đánh giá mức độ liên quan của context được cung cấp cho LLM.
- Hallucination Detection: Phát hiện xem câu trả lời có chứa thông tin không có trong context hay không.

Các framework đánh giá phổ biến: RAGAS, DeepEval, LangSmith.
"""

# ============================================================
# 10 CÂU HỎI - dài, gây nhiễu, đáp án dài
# ============================================================
QUESTIONS = [
    {
        "question": (
            "Trong kiến trúc ReAct Agent, quy trình lặp được mô tả gồm các bước chính nào? "
            "Hãy lưu ý rằng một số hệ thống agent khác như Plan-and-Execute cũng có quy trình lặp "
            "nhưng khác biệt ở chỗ tách bạch hoàn toàn giữa lập kế hoạch và thực thi. "
            "A. Planning → Execution → Replanning → Final Answer, trong đó agent lập kế hoạch tổng thể trước rồi thực thi tuần tự từng bước "
            "B. Thought → Action → Observation → Tiếp tục chu trình, trong đó mô hình suy luận, thực hiện hành động, quan sát kết quả rồi tiếp tục "
            "C. Query → Retrieval → Generation → Evaluation, trong đó hệ thống nhận câu hỏi, truy xuất tài liệu, sinh câu trả lời rồi đánh giá "
            "D. Input → Encoding → Decoding → Output, trong đó mô hình transformer mã hóa đầu vào thành biểu diễn ẩn rồi giải mã thành đầu ra"
        ),
        "correct": "B"
    },
    {
        "question": (
            "Trong hệ thống RAG (Retrieval-Augmented Generation), bước Indexing được mô tả là gì? "
            "Lưu ý rằng RAG gồm nhiều bước khác nhau từ Indexing, Retrieval đến Augmented Generation, "
            "và mỗi bước đều có vai trò riêng biệt trong pipeline tổng thể. "
            "A. Khi nhận được câu hỏi, hệ thống chuyển câu hỏi thành vector embedding và tìm kiếm các đoạn tài liệu có nội dung tương tự nhất trong vector database "
            "B. Các đoạn tài liệu liên quan được đưa vào prompt cùng với câu hỏi, và LLM sử dụng thông tin này để sinh câu trả lời chính xác hơn "
            "C. Tài liệu được chia thành các đoạn nhỏ (chunks), mỗi đoạn được chuyển thành vector embedding và lưu vào vector database "
            "D. Mô hình ngôn ngữ lớn được fine-tune trên tập dữ liệu chuyên ngành để cải thiện khả năng trả lời câu hỏi trong lĩnh vực cụ thể"
        ),
        "correct": "C"
    },
    {
        "question": (
            "Điểm khác biệt chính giữa mô hình chỉ có reasoning (suy luận) và ReAct trong bối cảnh AI agents là gì? "
            "Cần phân biệt rõ giữa các hạn chế của từng phương pháp, bao gồm cả mô hình chỉ có acting "
            "vốn có thể thực thi nhanh nhưng thiếu suy luận. "
            "A. Mô hình chỉ có reasoning có thể tương tác với thế giới bên ngoài và gọi công cụ, trong khi ReAct chỉ dựa vào kiến thức nội tại đã được huấn luyện "
            "B. Mô hình chỉ có reasoning dựa hoàn toàn vào khả năng ngôn ngữ của LLM, không thể tương tác với bên ngoài và dễ sinh ra câu trả lời sai, trong khi ReAct kết hợp cả suy luận và hành động qua vòng lặp Thought-Action-Observation "
            "C. Mô hình chỉ có reasoning sử dụng Plan-and-Execute để tách bạch hoàn toàn giữa lập kế hoạch và thực thi, trong khi ReAct chỉ thực hiện hành động đơn thuần "
            "D. Mô hình chỉ có reasoning hoạt động dựa trên Reciprocal Rank Fusion để kết hợp kết quả từ nhiều nguồn, trong khi ReAct sử dụng cosine similarity để tìm kiếm"
        ),
        "correct": "B"
    },
    {
        "question": (
            "Reranker trong hệ thống RAG có vai trò gì và có đặc điểm gì so với bi-encoder? "
            "Lưu ý rằng có nhiều thành phần khác trong RAG như embedding model, vector database, "
            "chunking strategy, và LLM, mỗi thành phần có chức năng riêng biệt. "
            "A. Reranker là mô hình bi-encoder dùng để tạo vector embedding cho văn bản, hoạt động song song với embedding model và nhanh hơn cross-encoder "
            "B. Reranker là cơ sở dữ liệu lưu trữ vector, cho phép tìm kiếm nhanh các đoạn văn bản có ngữ nghĩa tương tự với truy vấn của người dùng "
            "C. Reranker là chiến lược chia tài liệu thành các đoạn nhỏ theo ngữ nghĩa, giúp tăng chất lượng retrieval bằng cách đảm bảo mỗi chunk chứa một ý hoàn chỉnh "
            "D. Reranker là mô hình cross-encoder dùng để sắp xếp lại kết quả retrieval, giúp chọn ra các đoạn văn bản liên quan nhất, thường chính xác hơn bi-encoder nhưng chậm hơn"
        ),
        "correct": "D"
    },
    {
        "question": (
            "Trong NLP Pipeline, bước Named Entity Recognition (NER) có chức năng gì? "
            "Cần lưu ý rằng NLP Pipeline bao gồm nhiều bước như Sentence Segmentation, Tokenization, "
            "Part-of-Speech Tagging, Dependency Parsing và Coreference Resolution, mỗi bước xử lý một khía cạnh khác nhau. "
            "A. Tách đoạn văn thành các câu riêng biệt để phân tích từng câu một cách độc lập, đây là bước đầu tiên trong pipeline xử lý ngôn ngữ tự nhiên "
            "B. Tách câu thành các token bao gồm từ, sub-word hoặc ký tự, chuẩn bị cho các bước phân tích tiếp theo trong pipeline "
            "C. Nhận dạng và phân loại các thực thể có tên bao gồm người, tổ chức, địa điểm, ngày tháng trong văn bản đầu vào "
            "D. Phân tích cú pháp phụ thuộc để xác định mối quan hệ ngữ pháp giữa các từ trong câu, ví dụ chủ ngữ-vị ngữ"
        ),
        "correct": "C"
    },
    {
        "question": (
            "GraphRAG kết hợp Knowledge Graph và RAG cho phép thực hiện được những gì? "
            "Hãy phân biệt với RAG truyền thống chỉ sử dụng unstructured data và vector similarity search. "
            "Microsoft đã phát triển implementation cụ thể sử dụng community detection algorithms. "
            "A. GraphRAG chỉ hỗ trợ tìm kiếm từ khóa (sparse retrieval) thông qua BM25, không sử dụng vector embedding, và giới hạn trong các câu hỏi đơn giản một bước "
            "B. GraphRAG cho phép truy xuất thông tin chính xác hơn thông qua các mối quan hệ trong đồ thị, cung cấp context phong phú hơn cho LLM, hỗ trợ câu hỏi phức tạp nhiều bước suy luận, và giảm hallucination "
            "C. GraphRAG thay thế hoàn toàn LLM bằng hệ thống rule-based dựa trên đồ thị tri thức, không cần embedding model hay vector database trong pipeline "
            "D. GraphRAG chỉ hoạt động với dữ liệu có cấu trúc từ cơ sở dữ liệu quan hệ (SQL), không xử lý được văn bản phi cấu trúc hay dữ liệu tự nhiên"
        ),
        "correct": "B"
    },
    {
        "question": (
            "Phương pháp Hybrid Search kết hợp những loại tìm kiếm nào và sử dụng kỹ thuật gì để kết hợp kết quả? "
            "Trong ngữ cảnh RAG, việc kết hợp nhiều phương pháp retrieval là rất quan trọng "
            "để đảm bảo cả từ khóa chuyên ngành và ngữ nghĩa đều được xem xét. "
            "A. Chỉ sử dụng Dense retrieval (Vector Search) kết hợp với Cosine Similarity, không cần Sparse retrieval vì vector embedding đã đủ để bắt mọi ngữ nghĩa "
            "B. Kết hợp Sentence Segmentation và Tokenization để tạo ra các biểu diễn đa cấp độ cho mỗi tài liệu, sử dụng Part-of-Speech Tagging để xếp hạng "
            "C. Kết hợp Sparse retrieval (BM25) cho tìm kiếm từ khóa và Dense retrieval (Vector Search) cho tìm kiếm ngữ nghĩa, sử dụng Reciprocal Rank Fusion (RRF) để kết hợp kết quả "
            "D. Sử dụng Knowledge Graph traversal kết hợp với community detection algorithms của Microsoft GraphRAG để tìm kiếm đa chiều trong đồ thị tri thức"
        ),
        "correct": "C"
    },
    {
        "question": (
            "Trong kiến trúc Plan-and-Execute Agent, sự khác biệt chính so với ReAct Agent là gì? "
            "Cả hai đều là kiến trúc agent phổ biến nhưng có cách tiếp cận khác nhau. "
            "Cần phân biệt rõ ràng giữa việc suy luận xen kẽ hành động và việc lập kế hoạch trước rồi thực thi sau. "
            "A. Plan-and-Execute lập kế hoạch tổng thể trước rồi thực thi tuần tự, có tầm nhìn xa hơn nhưng kém linh hoạt hơn khi gặp tình huống bất ngờ, trong khi ReAct suy luận và hành động xen kẽ trong từng bước "
            "B. Plan-and-Execute sử dụng vòng lặp Thought-Action-Observation giống ReAct nhưng thêm bước Planning ở đầu, không có giai đoạn Replanning sau mỗi bước thực thi "
            "C. Plan-and-Execute dựa trên vector similarity search để lập kế hoạch, trong khi ReAct sử dụng knowledge graph để suy luận về mối quan hệ giữa các bước "
            "D. Plan-and-Execute chỉ hoạt động với các tác vụ đơn giản một bước, trong khi ReAct được thiết kế cho các tác vụ phức tạp đòi hỏi nhiều bước suy luận"
        ),
        "correct": "A"
    },
    {
        "question": (
            "Phương pháp Hypothetical Document Embedding (HyDE) hoạt động như thế nào trong Advanced Retrieval Strategies? "
            "Đây là một trong nhiều chiến lược retrieval nâng cao bao gồm Multi-query retrieval, Contextual compression, "
            "Parent-child chunking và Sentence Window Retrieval, mỗi chiến lược có ưu nhược điểm riêng. "
            "A. HyDE truy xuất chunk con nhỏ nhưng cung cấp chunk cha lớn hơn làm context, cho phép retrieval chính xác ở mức câu nhưng vẫn giữ được ngữ cảnh rộng hơn "
            "B. HyDE sinh nhiều biến thể câu hỏi khác nhau từ câu hỏi gốc để truy vấn, tăng coverage bằng cách đa dạng hóa góc nhìn truy vấn "
            "C. HyDE sinh document giả từ câu hỏi, sau đó dùng embedding của document giả đó để truy vấn tìm kiếm tài liệu tương tự trong vector database "
            "D. HyDE nén context để chỉ giữ lại thông tin liên quan nhất với câu hỏi, loại bỏ các phần thừa và nhiễu trong tài liệu đã truy xuất"
        ),
        "correct": "C"
    },
    {
        "question": (
            "Đánh giá hệ thống RAG bao gồm những khía cạnh nào và sử dụng các metric gì? "
            "Việc đánh giá toàn diện cần xem xét cả chất lượng retrieval, chất lượng câu trả lời, "
            "mức độ liên quan của context và khả năng phát hiện hallucination. "
            "A. Chỉ đánh giá Retrieval Quality thông qua Precision@K và Recall@K, không cần đánh giá chất lượng câu trả lời vì LLM đã được huấn luyện tốt "
            "B. Đánh giá bao gồm Retrieval Quality (Precision@K, Recall@K, MRR), Answer Quality (Faithfulness, Answer Relevancy, Correctness), Context Relevance và Hallucination Detection, sử dụng framework như RAGAS, DeepEval, LangSmith "
            "C. Chỉ sử dụng BLEU score và ROUGE score để so sánh câu trả lời sinh ra với câu trả lời tham chiếu, đây là phương pháp đánh giá duy nhất trong RAG "
            "D. Đánh giá thông qua community detection algorithms để phân tích cấu trúc đồ thị tri thức, không cần đánh giá trực tiếp chất lượng câu trả lời"
        ),
        "correct": "B"
    },
]

# ============================================================
# STATE
# ============================================================
students: dict = {}  # student_id -> {server_url, score, status, current_question, detail}

# ============================================================
# SCHEMAS
# ============================================================
class RegisterPayload(BaseModel):
    server_url: str


class EvaluatePayload(BaseModel):
    document_received: Optional[bool] = False

# ============================================================
# TEACHER ENDPOINTS
# ============================================================
@app.post("/api/v1/competition/register")
async def register(payload: RegisterPayload, x_student_id: str = Header(alias="X-Student-ID")):
    students[x_student_id] = {
        "server_url": payload.server_url,
        "score": 0.0,
        "status": "registered",
        "current_question": 0,
        "detail": [],
    }
    logger.info(f"[REGISTER] {x_student_id} -> {payload.server_url}")
    return {
        "message": "Đăng ký thành công!",
        "student_id": x_student_id,
        "server_url": payload.server_url,
    }


@app.post("/api/v1/competition/evaluate")
async def evaluate(
    payload: EvaluatePayload = EvaluatePayload(),
    x_student_id: str = Header(alias="X-Student-ID"),
):
    if x_student_id not in students:
        return {"status": "error", "message": "Chưa đăng ký!"}

    student = students[x_student_id]
    student["status"] = "evaluating"
    student["score"] = 0.0
    student["current_question"] = 0
    student["detail"] = []

    # Chạy evaluation trong thread riêng để không block
    thread = threading.Thread(
        target=_run_evaluation,
        args=(x_student_id, bool(payload.document_received)),
        daemon=True,
    )
    thread.start()

    return {
        "student_id": x_student_id,
        "status": "evaluating",
        "document_received": bool(payload.document_received),
        "message": "Đang bắt đầu quá trình thi...",
    }


def _run_evaluation(student_id: str, document_received: bool = False):
    student = students[student_id]
    server_url = student["server_url"].rstrip("/")

    # Step 1: Upload document
    if document_received:
        logger.info("[EVAL] Bỏ qua /upload vì document_received=true")
    else:
        logger.info(f"[EVAL] Gửi /upload tới {server_url}")
        try:
            resp = requests.post(
                f"{server_url}/upload",
                json={"doc_id": "mock_doc", "text": DOCUMENT_TEXT},
                timeout=120,
            )
            upload_result = resp.json()
            logger.info(f"[EVAL] Upload response: {upload_result}")
        except Exception as e:
            logger.error(f"[EVAL] Upload failed: {e}")
            student["status"] = "error"
            student["detail"].append({"upload_error": str(e)})
            return

    # Step 2: Ask 10 questions
    for i, q in enumerate(QUESTIONS):
        student["current_question"] = i + 1
        logger.info(f"[EVAL] Câu {i+1}/10: {q['question'][:60]}...")

        try:
            resp = requests.post(
                f"{server_url}/ask",
                json={"question": q["question"]},
                timeout=60,
            )
            result = resp.json()
            answer = result.get("answer", "").strip().upper()
            is_correct = answer == q["correct"]

            if is_correct:
                student["score"] += 1.0

            detail = {
                "question_num": i + 1,
                "student_answer": answer,
                "correct_answer": q["correct"],
                "is_correct": is_correct,
                "sources_count": len(result.get("sources", [])),
            }
            student["detail"].append(detail)
            logger.info(
                f"[EVAL] Câu {i+1}: answer={answer}, correct={q['correct']}, "
                f"{'ĐÚNG' if is_correct else 'SAI'}"
            )

        except Exception as e:
            logger.error(f"[EVAL] Câu {i+1} failed: {e}")
            student["detail"].append({
                "question_num": i + 1,
                "error": str(e),
                "is_correct": False,
            })

        time.sleep(0.5)  # Delay nhỏ giữa các câu

    student["status"] = "completed"
    logger.info(
        f"[EVAL] Hoàn thành! {student_id}: {student['score']}/{len(QUESTIONS)} điểm"
    )


@app.post("/api/v1/competition/reset")
async def reset(x_student_id: str = Header(alias="X-Student-ID")):
    if x_student_id in students:
        students[x_student_id]["score"] = 0.0
        students[x_student_id]["status"] = "registered"
        students[x_student_id]["current_question"] = 0
        students[x_student_id]["detail"] = []
    return {
        "status": "success",
        "message": f"Đã reset trạng thái cho sinh viên {x_student_id}",
    }


@app.get("/api/v1/competition/result")
async def result(x_student_id: str = Header(alias="X-Student-ID")):
    if x_student_id not in students:
        return {"status": "error", "message": "Chưa đăng ký!"}
    s = students[x_student_id]
    return {
        "student_id": x_student_id,
        "score": s["score"],
        "status": s["status"],
        "current_question": s["current_question"],
        "detail": s["detail"],
    }


# ============================================================
# PROXY LLM (mock - trả lời đơn giản cho test)
# ============================================================
@app.post("/api/v1/proxy/chat/completions")
async def proxy_llm(request: Request):
    """Mock LLM proxy - trả lời dựa trên content trong messages."""
    body = await request.json()
    messages = body.get("messages", [])

    # Lấy nội dung user message cuối
    user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    # Mock: cố gắng trích xuất đáp án đúng từ context
    # Trong thực tế, đây sẽ là proxy tới OpenAI API
    import re
    answer = "A"  # Default fallback

    # Tìm kiếm pattern đáp án trong câu hỏi
    # Thử match các đáp án với context
    if "Thought → Action → Observation" in user_content and "Thought" in user_content:
        answer = "B"
    elif "Indexing" in user_content and "chunks" in user_content and "vector embedding" in user_content:
        answer = "C"
    elif "reasoning" in user_content.lower() and "acting" in user_content.lower() and "không thể tương tác" in user_content:
        answer = "B"
    elif "Reranker" in user_content and "cross-encoder" in user_content:
        answer = "D"
    elif "Named Entity Recognition" in user_content:
        answer = "C"
    elif "GraphRAG" in user_content and "mối quan hệ" in user_content:
        answer = "B"
    elif "Hybrid Search" in user_content and "BM25" in user_content:
        answer = "C"
    elif "Plan-and-Execute" in user_content and "tầm nhìn" in user_content:
        answer = "A"
    elif "HyDE" in user_content and "document giả" in user_content:
        answer = "C"
    elif "Retrieval Quality" in user_content and "RAGAS" in user_content:
        answer = "B"

    return {
        "id": "mock-completion",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    logger.info(f"Mock Teacher Server starting on {TEACHER_HOST}:{TEACHER_PORT}")
    logger.info(f"Student endpoints: /api/v1/competition/[register|evaluate|reset|result]")
    logger.info(f"LLM proxy: /api/v1/proxy/chat/completions")
    uvicorn.run(app, host=TEACHER_HOST, port=TEACHER_PORT, log_level="info")

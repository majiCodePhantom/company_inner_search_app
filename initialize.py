"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
import docx
from langchain.schema import Document as LCdocument # for text-based documents
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct
#pandas追加
import pandas as pd



############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
 try:
        initialize_session_state()
        print("init: session_state OK")
        initialize_session_id()
        print("init: session_id OK")
        initialize_logger()
        print("init: logger OK")
        initialize_retriever()
        print("init: retriever OK")
 except Exception as e:
        # 本番でも必ず見えるところに出す（Streamlitのログ/コンソール）
        import traceback, logging
        logging.exception("initialize() failed: %s", e)
        # 開発時だけUIにも
        if os.getenv("APP_ENV","prod") != "prod":
            st.sidebar.error("initialize() で例外発生")
            st.sidebar.code("".join(traceback.format_exc()))
        raise

    # """
    # 画面読み込み時に実行する初期化処理
    # """
    # # 初期化データの用意
    # initialize_session_state()
    # # ログ出力用にセッションIDを生成
    # initialize_session_id()
    # # ログ出力の設定
    # initialize_logger()
    # # RAGのRetrieverを作成
    # initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意 モデル変えてみる
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.DEFAULT_CHUNK_SIZE,
        chunk_overlap=ct.DEFAULT_CHUNK_OVERLAP,
        separator="\n"
    )

    # チャンク分割を実施
    splitted_docs = text_splitter.split_documents(docs_all)

    # ベクターストアの作成
    db = Chroma.from_documents(splitted_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RETRIEVAL_FETCH_K})


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []




##############################################################################3
def load_data_sources():
    docs_all = []

    # --- ローカルファイル ---
    for root, _, files in os.walk(ct.RAG_TOP_FOLDER_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ct.SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(root, file)
            loader = ct.SUPPORTED_EXTENSIONS[ext](file_path)
            docs = loader.load()

            # ★ PDFだけページ番号を追加
            if ext == ".pdf":
                for i, d in enumerate(docs):
                    d.metadata["page_number"] = int(d.metadata.get("page", i)) + 1

            docs_all.extend(docs)
#########################################csvファイル精度向上対策#######################
            if ext == ".csv":
                # CSVファイルの場合、pandasで読み込み　デバッグ追加

                log = logging.getLogger(ct.LOGGER_NAME)
                
                df = pd.read_csv(file_path, encoding="utf-8", dtype=str, keep_default_na=False)
                df.columns = df.columns.str.strip()  # 前後空白除去
                log.info("CSV columns (normalized) for %s: %s", file_path, list(df.columns))
                # 各行をドキュメントとして扱う
                for i, row in df.iterrows():
                    text = f"社員ID: {row['社員ID']}, 氏名（フルネーム）: {row['氏名（フルネーム）']}, 性別: {row['性別']}, 生年月日: {row['生年月日']}, 年齢: {row['年齢']}, メールアドレス: {row['メールアドレス']}, 従業員区分: {row['従業員区分']}, 入社日: {row['入社日']}, 部署: {row['部署']}, 役職:{row['役職']}, スキルセット: {row['スキルセット']}, 保有資格: {row['保有資格']}, 大学名: {row['大学名']}, 学部・学科: {row['学部・学科']},卒業年月日: {row['卒業年月日']}"
                    
                    docs.append(LCdocument(
                        page_content=text,
                        metadata={"row_id": i, "source": file_path}
                    ))
                docs_all.extend(docs)
                


######################################################################

    # --- Webページ ---
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        docs_all.extend(WebBaseLoader(web_url).load())

    return docs_all
############################################################################################















# def load_data_sources():
#     """
#     RAGの参照先となるデータソースの読み込み

#     Returns:
#         読み込んだ通常データソース
#     """
#     # データソースを格納する用のリスト
#     docs_all = []
#     # ファイル読み込みの実行（渡した各リストにデータが格納される）
#     recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

#     web_docs_all = []
#     # ファイルとは別に、指定のWebページ内のデータも読み込み
#     # 読み込み対象のWebページ一覧に対して処理
#     for web_url in ct.WEB_URL_LOAD_TARGETS:
#         # 指定のWebページを読み込み
#         loader = WebBaseLoader(web_url)
#         web_docs = loader.load()
#         # for文の外のリストに読み込んだデータソースを追加
#         web_docs_all.extend(web_docs)
#     # 通常読み込みのデータソースにWebページのデータを追加
#     docs_all.extend(web_docs_all)

#     return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s
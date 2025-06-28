import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--retrieveOnly", action="store_true", help="只進行向量資料庫檢索，不啟用 LLM")
args = parser.parse_args()
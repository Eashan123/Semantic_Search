from typing import AnyStr, List


class Dataset:
    def __init__(self, file_path: AnyStr):
        self.file_path = file_path
        self.data = self._read_from_file()
        self.len = len(self.data)

    def _read_from_file(self) -> List[dict]:
        with open(self.file_path, "r", errors="ignore") as f:
            return [
                {"id": row_id, "text": row.replace("\n", "")}
                for row_id, row in enumerate(f.readlines())
            ]

    def get_documents_by_id(self, doc_ids: List[int]) -> List[str]:
        return [self.data[doc_id]["text"] for doc_id in doc_ids]

    def get_documents(self, n: int = -1) -> dict:
        for i, row in enumerate(self.data):
            if i == n:
                break
            yield row

    def __len__(self):
        return self.len

    def __str__(self):
        return (
            f"I am a collection of {self.len} descriptions from path {self.file_path}"
        )

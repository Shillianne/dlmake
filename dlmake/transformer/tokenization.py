from typing import Optional, Collection

class Tokenizer:
    def __init__(self, vocabulary: Collection[str], special_tokens: Optional[Collection[str]] = None):
        self.enc = {token: i for i, token in zip(range(len(vocabulary)), vocabulary)}
        self.dec = {i: token for i, token in zip(range(len(vocabulary)), vocabulary)}
        self._special_tokens: dict[str, int] = {}
        if special_tokens is not None:
            for i, s_token in enumerate(special_tokens): 
                self.enc[s_token] = max(self.enc.values()) + i + 1
                self.dec[max(self.enc.values()) + i + 1] = s_token
                self._special_tokens[s_token] = self.enc[s_token]

    def encode(self, sequence: list) -> list:
        try:
            return [self.enc[token] for token in sequence]
        except KeyError as e:
            raise e

    def decode(self, sequence: list) -> list:
        try:
            return [self.dec[token] for token in sequence]
        except KeyError as e:
            raise e

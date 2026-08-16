import string

import sentencepiece_model_pb2


class SentencePieceProcessor:
    def __init__(self):
        """Process the tokenizer model file and get the usefull infos"""

        model = sentencepiece_model_pb2.ModelProto()
        with open("tokenizer.model", "rb") as f:
            model.ParseFromString(f.read())

        self.fields: list[str] = []
        for field in model.DESCRIPTOR.fields:
            self.fields.append(field.name)
            setattr(self, field.name, getattr(model, field.name))

        self.p: list[str] = []  # pieces list
        self.s: list[float] = []  # scores list
        self.ids: dict[str, int] = {}
        for id, v in enumerate(model.pieces):
            self.p.append(v.piece)
            self.s.append(v.score)
            self.ids[v.piece] = id

    def inspect(self):
        """print a simple list of the structure of the tokenizer.model file"""

        for field in self.fields:
            if field == "pieces":
                continue

            print("-----------------------")
            print("**", field, "**")
            print(getattr(self, field))

    def encode():
        """raw text -> normalize -> BPE encoding -> byte fallback -> token IDs"""

    def decode():
        """token IDs -> byte handling -> text"""


sp = SentencePieceProcessor()
sp.inspect()

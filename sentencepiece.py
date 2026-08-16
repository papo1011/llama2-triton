import string

import sentencepiece_model_pb2


class SentencePieceProcessor:
    def __init__(self):
        """Process the tokenizer model file and get the usefull infos"""
        model = sentencepiece_model_pb2.ModelProto()
        with open("tokenizer.model", "rb") as f:
            model.ParseFromString(f.read())

        self.pieces: list[string] = []
        self.scores: list[float] = []
        self.ids: dict[string, int] = {}
        for id, v in enumerate(model.pieces):
            self.pieces.append(v.piece)
            self.scores.append(v.score)
            self.ids[v.piece] = id

    def encode():
        """raw text -> normalize -> BPE encoding -> byte fallback -> token IDs"""

    def decode():
        """token IDs -> byte handling -> text"""


sp = SentencePieceProcessor()

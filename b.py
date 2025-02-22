MultiOutputPattern(
    [
        CallFunction(
            aten.mul.Tensor,
            CallFunction(operator.getitem, CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2), 0, _users=2),
            CallFunction(
                aten.mul.Tensor,
                CallFunction(
                    operator.getitem, CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2), 1, _users=4
                ),
                CallFunction(
                    aten.sigmoid.default,
                    CallFunction(
                        operator.getitem, CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2), 1, _users=4
                    ),
                ),
                _users=2,
            ),
        ),
        CallFunction(
            aten.cat.default,
            [
                CallFunction(
                    aten.mul.Tensor,
                    KeywordArg("tangents_1"),
                    CallFunction(
                        aten.mul.Tensor,
                        CallFunction(
                            operator.getitem, CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2), 1, _users=4
                        ),
                        CallFunction(
                            aten.sigmoid.default,
                            CallFunction(
                                operator.getitem,
                                CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2),
                                1,
                                _users=4,
                            ),
                        ),
                        _users=2,
                    ),
                ),
                CallFunction(
                    aten.mul.Tensor,
                    CallFunction(
                        aten.mul.Tensor,
                        KeywordArg("tangents_1"),
                        CallFunction(
                            operator.getitem, CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2), 0, _users=2
                        ),
                    ),
                    CallFunction(
                        aten.mul.Tensor,
                        CallFunction(
                            aten.sigmoid.default,
                            CallFunction(
                                operator.getitem,
                                CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2),
                                1,
                                _users=4,
                            ),
                            _users=2,
                        ),
                        CallFunction(
                            aten.add.Scalar,
                            CallFunction(
                                aten.mul.Tensor,
                                CallFunction(
                                    operator.getitem,
                                    CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2),
                                    1,
                                    _users=4,
                                ),
                                CallFunction(
                                    aten.sub.Tensor,
                                    CallFunction(aten.full.default, layout=torch.strided, pin_memory=False),
                                    CallFunction(
                                        aten.sigmoid.default,
                                        CallFunction(
                                            operator.getitem,
                                            CallFunction(aten.split.Tensor, KeywordArg("x"), _users=2),
                                            1,
                                            _users=4,
                                        ),
                                        _users=2,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ],
        ),
    ]
)

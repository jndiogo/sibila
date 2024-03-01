---
title: Threads, Messages, Context
---


::: sibila.Thread
    options:
        members:
            - __init__
            - clear
            - last_kind
            - last_text
            - inst
            - add
            - addx
            - get_text
            - set_text
            - concat
            - load
            - save
            - init_inst_in
            - add_in
            - add_out
            - add_out_in
            - make_inst_in
            - make_out_in
            - msg_as_chatml
            - as_chatml
            - has_text_lower


::: sibila.MsgKind
    options:
        members:
            - IN
            - OUT
            - INST

::: sibila.Context
    options:
        members:
            - __init__
            - clear
            - trim

::: sibila.Trim


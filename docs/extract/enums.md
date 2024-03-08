---
title: Enums
---

Enumerations are important for classification tasks or in any situation where you need a choice to be made from a list of options.

!!! example
    ``` python
    from sibila import Models

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    model.extract(["red", "blue", "green", "yellow"], 
                  "The car color was a shade of indigo")
    ```

    !!! success "Result"
        ```
        'blue'
        ```
You can pass a list of items in any of the supported native types: str, float, int or bool.


## Literals

We can also use Literals:

!!! example
    ``` python
    from typing import Literal

    model.extract(Literal["SPAM", "NOT_SPAM", "UNSURE"], 
                 "Hello my dear friend, I'm contacting you because I want to give you a million dollars",
                 inst="Classify this text on the likelihood of being spam")
    ```

    !!! success "Result"
        ```
        'SPAM'
        ```

Extracting to a Literal type returns one of its possible options in its native type (str, float, int or bool).


## Enum classes

Or Enum classes of native types. An example of extracting to Enum classes:

!!! example
    ``` python
    from enum import IntEnum

    class Heads(IntEnum):
        SINGLE = 1
        DOUBLE = 2
        TRIPLE = 3

    model.extract(Heads,
                  "The Two-Headed Monster from The Muppets.")
    ```

    !!! success "Result"
        ```
        <Heads.DOUBLE: 2>
        ```


For the model, the important information is actual the value of each enum member, not its name. For example, in this enum, the model would only see the strings to the right of each member (the enum values), not "RED", "ORANGE" nor "GREEN":

``` python
class Light(Enum):
    RED = 'stop'
    YELLOW = 'slow down'
    GREEN = 'go'
```

See the [Tag classification example](../examples/tag.md) to see how Enum is used to tag support queries.



## Classify

You can also use the [classify()](../api-reference/model.md#sibila.LlamaCppModel.classify) method to extract enumerations, which accepts the enum types we've seen above. It calls extract() internally and its only justification is to make things more readable:


!!! example
    ``` python
    model.classify(["mouse", "cat", "dog", "bird"],
                   "Snoopy")
    ```

    !!! success "Result"
        ```
        'dog'
        ```

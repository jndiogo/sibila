---
title: Simple types
---

Sibila can constrain model generation to output simple python types. This is helpful for situations where you want to extract a specific data type. 

To get a response from the model in a certain type, you can use the [extract()](../api-reference/local_model.md#sibila.Model.extract) method:

!!! example
    ``` python
    from sibila import Models

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    model.extract(bool, 
                  "Certainly, I'd like to subscribe.")
    ```

    !!! success "Result"
        ```
        True
        ```

## Instructions to help the model

You may need to provide more extra information to the model, so that it understands what you want. This is done with the inst argument - inst is a shorter name for instructions:

!!! example
    ``` python
    model.extract(str, 
                "I don't quite remember the product's name, I think it was called Cornaca",
                inst="Extract the product name")
    ```

    !!! success "Result"
        ```
        Cornaca
        ```


## Supported types

The following simple types are supported:

- bool
- int
- float
- str
- datetime



!!! note "About datetime type"

    A special note about extracting to datetime: the datetime type is expecting an ISO 8601 formatted string. Because some models are less capable than others at correctly formatting dates/times, it helps to mention in the instructions that you want the output in "ISO 8601" format.

    
    ``` python
    from datetime import datetime
    model.extract(datetime, 
                  "Sure, glad to help, it all happened at December the 10th, 2023, around 3PM, I think",
                  inst="Output in ISO 8601 format")
    ```

    !!! success "Result"
        ```
        datetime.datetime(2023, 12, 10, 15, 0)
        ```




## Lists

You can extract lists of any of the supported types (simple types, enum, dataclass, Pydantic).

!!! example
    ``` python
    model.extract(list[str], 
                 "I'd like to visit Naples, Genoa, Florence and of course, Rome")
    ```

    !!! success "Result"
        ```
        ['Naples', 'Genoa', 'Florence', 'Rome']
        ```

As in all extractions, you may need to set the instructions text to specify what you want from the model. Just as an example of the power of instructions, let's add instructions asking for country output: it will still output a list, but with a single element - 'Italy':

!!! example
    ``` python
    model.extract(list[str], 
                 "I'd like to visit Naples, Genoa, Florence and of course, Rome",
                 inst="Output the country")
    ```

    !!! success "Result"
        ```
        ['Italy']
        ```


---
title: Free text
---

You can also generate plain text by calling [Model()](../api-reference/local_model.md#sibila.Model.__call__) or [Model.call()](../api-reference/local_model.md#sibila.Model.call):


!!! example
    ``` python
    from sibila import Models

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    response = model("Explain in a few lines how to build a brick wall?")
    print(response)
    ```

    !!! success "Result"
        ``` text
        To build a brick wall, follow these steps:

        1. Prepare the site by excavating and leveling the ground, then install a damp-proof 
        membrane and create a solid base with concrete footings.
        2. Lay a foundation of concrete blocks or bricks, ensuring it is level and square.
        3. Build the wall using bricks or blocks, starting with a corner or bonding pattern 
        to ensure stability. Use mortar to bond each course (row) of bricks or blocks, 
        following the recommended mortar mix ratio.
        4. Use a spirit level to ensure each course is level, and insert metal dowels or use 
        brick ties to connect adjacent walls or floors.
        5. Allow the mortar to dry for the recommended time before applying a damp-proof 
        course (DPC) at the base of the wall.
        6. Finish the wall with capping bricks or coping stones, and apply any desired 
        render or finish.
        ```

---
title: Free JSON
---

Methods like [extract()](../api-reference/local_model.md#sibila.Model.extract) will generate JSON format constrained to a certain JSON Schema: this is needed or the model might not generate the fields or data types we're looking for.

You can generate schema-free JSON with the [json()](../api-reference/local_model.md#sibila.Model.json) method. In this case, the model will pick the field names and data types for you.

For example:

!!! example
    ``` python
    from sibila import Models

    Models.setup("../models")
    model = Models.create("llamacpp:openchat")

    response = model("How to build a brick wall?")

    from pprint import pprint
    pprint(response, sort_dicts=False)    
    ```

    !!! success "Result"
        ``` json
        {'steps': [{'step_number': 1,
                    'description': 'Gather all necessary materials and tools including '
                                'bricks, mortar, trowel, spirit level, tape '
                                'measure, bricklaying line, and safety equipment.'},
                {'step_number': 2,
                    'description': 'Prepare the foundation for the wall. Ensure it is '
                                'solid, level, and has the correct dimensions for '
                                'the wall you are building.'},
                {'step_number': 3,
                    'description': "Mix the mortar according to the manufacturer's "
                                'instructions, ensuring a consistent and workable '
                                'consistency.'},
                {'step_number': 4,
                    'description': 'Lay a bed of mortar where the first row of bricks '
                                'will be placed. Use the trowel to spread the '
                                'mortar evenly.'},
                {'step_number': 5,
                    'description': 'Start laying the bricks from one end, applying '
                                'mortar to the end of each brick before placing it '
                                'down to bond with the next brick.'},
                {'step_number': 6,
                    'description': 'Use the spirit level to check that the bricks are '
                                'level both horizontally and vertically. Adjust as '
                                'necessary.'},
                {'step_number': 7,
                    'description': 'Continue laying bricks, ensuring that you stagger '
                                'the joints in each row (running bond pattern). '
                                'This adds strength to the wall.'},
                {'step_number': 8,
                    'description': 'Periodically check that the wall is straight and '
                                'level by using the spirit level and the '
                                'bricklaying line.'},
                {'step_number': 9,
                    'description': 'Remove any excess mortar with the trowel as you '
                                'work. Keep the work area clean.'},
                {'step_number': 10,
                    'description': 'As you reach the end of each row, you may need to '
                                'cut bricks to fit. Use a brick hammer or a brick '
                                'cutter to do this.'},
                {'step_number': 11,
                    'description': 'Once the wall reaches the desired height, finish '
                                'the top with a row of solid bricks or capping '
                                'stones to protect the wall from weather.'},
                {'step_number': 12,
                    'description': 'Cure the mortar by protecting the wall from '
                                'extreme weather conditions for at least 24-48 '
                                'hours.'},
                {'step_number': 13,
                    'description': 'Clean the finished wall with a brush and water to '
                                'remove any remaining mortar residue.'},
                {'step_number': 14,
                    'description': 'Dispose of any waste material responsibly and '
                                'clean your tools.'}],
        'safety_tips': ['Wear safety glasses to protect your eyes from flying debris.',
                        'Use gloves to protect your hands from sharp edges and wet '
                        'mortar.',
                        'Wear a dust mask when mixing mortar to avoid inhaling dust '
                        'particles.',
                        'Keep the work area clear to prevent tripping hazards.'],
        'tools_required': ['Bricks',
                            'Mortar',
                            'Trowel',
                            'Spirit level',
                            'Tape measure',
                            'Bricklaying line',
                            'Safety glasses',
                            'Gloves',
                            'Dust mask',
                            'Brick hammer or cutter']}
        ```

The model returned a Python dictionary with fields and data types of it's own choice. We could provide a JSON Schema t defines a structure for the response.

See the [From text to object](../examples/from_text_to_object.md) example for a related use.
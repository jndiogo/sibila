# Chat templates

All templates listed can be applied with:

``` python
jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True,
                                          lstrip_blocks=True)
jinja_compiled_template = jinja_env.from_string(format_template)

text = jinja_compiled_template.render(messages=messages,
                                      add_generation_prompt=True,
                                      **{"bos_token": "...",
                                         "eos_token": ".."})
```




# Text Models


## ChatML

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'].strip() + '<|im_end|>' + '\n'}}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```




## Llama-3 Instruct

https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF


```
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = '<|start_header_id|>' + 'system' + '<|end_header_id|>\n\n' + messages[0]['content'].strip() + '<|eot_id|>' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}

  {% if loop.index0 == 0 %}
    {{ system_message }}
  {% endif %}
  
  {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}

  {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
  {% endif %}

{% endfor %}
```


# Mistral


```
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}

  {% if loop.index0 == 0 %}
    {% set content = system_message + message['content'] %}
  {% else %}
    {% set content = message['content'] %}
  {% endif %}
  
  {% if message['role'] == 'user' %}
    {{ '[INST] ' + content.strip() + ' [/INST]' }}
  {% elif message['role'] == 'assistant' %}
    {{ content.strip() + eos_token}}
  {% endif %}

{% endfor %}"
```



# Vicuna


```
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{{ system_message }}

{% for message in loop_messages %}
  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}
  
  {% if message['role'] == 'user' %}
    {{ 'USER: ' + message['content'].strip() + '\n' }}
  
  {% elif message['role'] == 'assistant' %}
  
    {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}
  {% endif %}
  
{% endfor %}

{% if add_generation_prompt %}
  {{ 'ASSISTANT:' }}
{% endif %}
```



## Phi3

https://huggingface.co/microsoft/Phi-3-mini-4k-instruct


```
{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'system') %}
        {{'<|system|>' + '\n' + message['content'].strip() + '<|end|>' + '\n'}}
    {% elif (message['role'] == 'user') %}
        {{'<|user|>' + '\n' + message['content'].strip() + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}
    {% elif message['role'] == 'assistant' %}
        {{message['content'].strip() + '<|end|>' + '\n'}}
    {% endif %}
{% endfor %}"
```










# Text + Vision Models


## Llava 1.5

https://huggingface.co/mys/ggml_llava-v1.5-7b


```
{% for message in messages %}
  {% if message.role == 'system' %}
    {{ message.content.strip() }}
    {{ '\n' }}

  {% elif message.role == 'user' %}
    {% if message.content is string %}
      USER: {{ message.content.strip() }}

    {% elif message.content is iterable %}
      USER: 
      
      {% for content in message.content %}
        
        {% if content.type == 'image_url' and content.image_url is mapping and content.image_url.url is string%}
          {{ content.image_url.url.strip() + ' ' }}
        {% endif %}

      {% endfor %}

      {% for content in message.content %}

        {% if content.type == 'text' %}
          {{ content.text.strip() }}
        {% endif %}

      {% endfor %}

    {% endif %}

    {{ '\n' }}

  {% elif message.role == 'assistant' and message.content is not none %}
    ASSISTANT: {{ message.content.strip() }}
    {{ '\n' }}
  {% endif %}

{% endfor %}

{% if add_generation_prompt %}
  ASSISTANT: 
{% endif %}
```




## Llava 1.6 Mistral 7B

https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf


```
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + ' ' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}

  {% if message.role == 'user' %}

    [INST]
  
    {% if message.content is string %}

      {% if system_message != '' %}
        {% set text = system_message + message.content.strip() %}
        {% set system_message = '' %}
      {% else %}
        {% set text = message.content.strip() %}
      {% endif %}

      {{ text }}

    {% elif message.content is iterable %}

      {% for content in message.content %}
        {% if content.type == 'image_url' and content.image_url is mapping %}
          {{ content.image_url.url.strip() + '\n' }}
        {% endif %}
      {% endfor %}

      {% for content in message.content %}
        {% if content.type == 'text' %}

          {% if system_message != '' %}
            {% set text = system_message + content.text.strip() %}
            {% set system_message = '' %}
          {% else %}
            {% set text = content.text.strip() %}
          {% endif %}

          {{ text }}
        {% endif %}
      {% endfor %}

    {% endif %}

    [/INST]

  {% elif message.role == 'assistant' %}
    {{ message.content.strip() }}
  {% endif %}

{% endfor %}

{% if add_generation_prompt %}
{% endif %}
```



## Llava 1.6 Vicuna 7B/13B

https://huggingface.co/cjpais/llava-v1.6-vicuna-7b-gguf


```
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + ' ' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if message.role == 'user' %}
    USER:{{ '\n' }}
  
    {% if message.content is string %}

      {% if system_message != '' %}
        {% set text = system_message + message.content.strip() %}
        {% set system_message = '' %}
      {% else %}
        {% set text = message.content.strip() %}
      {% endif %}

      {{ text }}

    {% elif message.content is iterable %}

      {% for content in message.content %}
        {% if content.type == 'image_url' and content.image_url is mapping %}
          {{ content.image_url.url + ' ' }}
        {% endif %}
      {% endfor %}

      {% for content in message.content %}
        {% if content.type == 'text' %}

          {% if system_message != '' %}
            {% set text = system_message + content.text.strip() %}
            {% set system_message = '' %}
          {% else %}
            {% set text = content.text.strip() %}
          {% endif %}

          {{ text }}
        {% endif %}
      {% endfor %}

    {% endif %}

    {{ '\n' }}

  {% elif message.role == 'assistant' %}
    ASSISTANT:{{ '\n' + message.content + '\n' }}
  {% endif %}

{% endfor %}

{% if add_generation_prompt %}
  ASSISTANT:{{ '\n' }}
{% endif %}
```



## Llava 1.6 Hermes 34B

Based on Nous-Hermes-2-Yi-34B.

https://huggingface.co/cjpais/llava-v1.6-34B-gguf


```
{% for message in messages %}

  {% if message.role == 'user' %}
    {{ '<|im_start|>user\n' }}

    {% if message.content is string %}
        {{ message.content.strip() }}

    {% elif message.content is iterable %}
      
      {% for content in message.content %}
        
        {% if content.type == 'image_url' and content.image_url is mapping and content.image_url.url is string%}
          {{ content.image_url.url.strip() + '\n' }}
        {% endif %}

      {% endfor %}

      {% for content in message.content %}

        {% if content.type == 'text' %}
          {{ content.text.strip() }}
        {% endif %}

      {% endfor %}

    {% endif %}

    {{ '<|im_end|>' + '\n' }}

  {% elif message.role == 'assistant' or message.role == 'system' and message.content is not none %}
    {{ '<|im_start|>' + message.role + '\n' + message.content.strip() + '<|im_end|>' + '\n' }}
  {% endif %}

{% endfor %}

{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```











## Moondream2

https://huggingface.co/vikhyatk/moondream2


```
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = messages[0]['content'].strip() + '\n' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if message.role == 'user' %}
  
    {% if message.content is string %}

      {% if system_message != '' %}
        {% set text = system_message + message.content.strip() %}
        {% set system_message = '' %}
      {% else %}
        {% set text = message.content.strip() %}
      {% endif %}

      Question: {{ text + '\n\n' }}

    {% elif message.content is iterable %}

      {% for content in message.content %}
        {% if content.type == 'image_url' and content.image_url is mapping %}
          {{ content.image_url.url + '\n\n' }}
        {% endif %}
      {% endfor %}

      {% for content in message.content %}
        {% if content.type == 'text' %}

          {% if system_message != '' %}
            {% set text = system_message + content.text.strip() %}
            {% set system_message = '' %}
          {% else %}
            {% set text = content.text.strip() %}
          {% endif %}

          Question: {{ text + '\n\n' }}
        {% endif %}
      {% endfor %}

    {% endif %}

  {% elif message.role == 'assistant' %}
    Answer: {{ message.content + '\n\n' }}
  {% endif %}

{% endfor %}

{% if add_generation_prompt %}
  Answer: 
{% endif %}
```


## LLava-phi3

https://huggingface.co/xtuner/llava-phi-3-mini-gguf


{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'system') %}
        {{'<|system|>' + '\n' + message['content'].strip() + '<|end|>' + '\n'}}
    {% elif (message['role'] == 'user') %}
        {{'<|user|>' + '\n' + message['content'].strip() + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}
    {% elif message['role'] == 'assistant' %}
        {{message['content'].strip() + '<|end|>' + '\n'}}
    {% endif %}
{% endfor %}"

-------------------------

```
{{ bos_token }}
{% for message in messages %}
  {% if message.role == 'system' %}
    {{'<|system|>' + '\n' + message.content.strip() + '<|end|>' + '\n'}}

  {% elif message.role == 'user' %}
    {{ '<|user|>' + '\n' }}

    {% if message.content is string %}
      {{ message.content.strip() }}

    {% elif message.content is iterable %}
      
      {% for content in message.content %}
        
        {% if content.type == 'image_url' and content.image_url is mapping and content.image_url.url is string%}
          {{ content.image_url.url.strip() + '\n' }}
        {% endif %}

      {% endfor %}

      {% for content in message.content %}

        {% if content.type == 'text' %}
          {{ content.text.strip() }}
        {% endif %}

      {% endfor %}

    {% endif %}

    {{ '<|end|>' + '\n' + '<|assistant|>' }}

  {% elif message.role == 'assistant' and message.content is not none %}
    {{ message['content'].strip() + '<|end|>' + '\n' }}

  {% endif %}

{% endfor %}
```








## Llava-Llama-3

https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf


```
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = '<|start_header_id|>' + 'system' + '<|end_header_id|>\n\n' + messages[0]['content'].strip() + '<|eot_id|>' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}

  {% if loop.index0 == 0 %}
    {{ system_message }}
  {% endif %}
  
  {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
  
  {% if message.content is string %}
    {{ message.content.strip() }}

  {% elif message.content is iterable %}
    
    {% for content in message.content %}
      
      {% if content.type == 'image_url' and content.image_url is mapping and content.image_url.url is string%}
        {{ content.image_url.url + '\n' }}
      {% endif %}

    {% endfor %}

    {% for content in message.content %}

      {% if content.type == 'text' %}
        {{ content.text.strip() }}
      {% endif %}

    {% endfor %}

  {% endif %}

  {{ '<|eot_id|>' }}

  {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
  {% endif %}

{% endfor %}
```



## Llama-3-vision-alpha

https://huggingface.co/qresearch/llama-3-vision-alpha-hf


```
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
  {% set loop_messages = messages[1:] %}
  {% set system_message = '<|start_header_id|>' + 'system' + '<|end_header_id|>\n\n' + messages[0]['content'].strip() + '<|eot_id|>' %}
{% else %}
  {% set loop_messages = messages %}
  {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}

  {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
  {% endif %}

  {% if loop.index0 == 0 %}
    {{ system_message }}
  {% endif %}
  
  {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
  
  {% if message.content is string %}
    {{ message.content.strip() }}

  {% elif message.content is iterable %}
    
    {% for content in message.content %}
      
      {% if content.type == 'image_url' and content.image_url is mapping and content.image_url.url is string%}
        {{ content.image_url.url + '\n' }}
      {% endif %}

    {% endfor %}

    {% for content in message.content %}

      {% if content.type == 'text' %}
        {{ content.text.strip() }}
      {% endif %}

    {% endfor %}

  {% endif %}

  {{ '<|eot_id|>' }}

  {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
  {% endif %}

{% endfor %}
```

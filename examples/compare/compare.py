# load env variables like OPENAI_API_KEY from a .env file (if available)
try: from dotenv import load_dotenv; load_dotenv()
except: ...

if __name__ == "__main__":

    from sibila import Models

    # to use a local model, assuming it's in ../../models:
    # setup models folder:
    Models.setup("../../models")
    # set the model's filename - change to your own model
    local_name = "llamacpp:openchat-3.5-1210.Q4_K_M.gguf"

    # to use an OpenAI model:
    remote_name = "openai:gpt-3.5"


    reviews = [
    "The user manual was confusing, but once I figured it out, the product more or less worked.",
    "This widget changed my life! It's sleek, efficient, and worth every penny.",
    "I'm disappointed with the product quality. It broke after just a week of use.",
    "The customer service team was incredibly helpful in resolving my issue with the device.",
    "I'm blown away by the functionality of this gadget. It exceeded my expectations.",
    "The packaging was damaged upon arrival, but the product itself works great.",
    "I've been using this tool for months, and it's still as good as new. Highly recommended!",
    "I regret purchasing this item. It doesn't perform as advertised.",
    "I've never had so much trouble with a product before. It's been a headache from day one.",
    "I bought this as a gift for my friend, and they absolutely love it!",
    "The price seemed steep at first, but after using it, I understand why. Quality product.",
    "This gizmo is a game-changer for my daily routine. Couldn't be happier with my purchase!"
    ]


    from sibila.multigen import (
        query_multigen,
        make_extract_gencall
    )

    # model instructions text, also known as system message
    inst_text = "You are a helpful assistant that analyses text sentiment."

    # the input query, including the above review lines
    in_text = "Each line is a product review. Extract the sentiment associated with each review:\n\n" + reviews[0]

    sentiment_enum = ["positive", "neutral", "negative"]

    out = query_multigen(reviews,
                         inst_text,
                         model_names = [local_name, remote_name],
                         text="print",
                         csv="sentiment.csv",
                         out_keys = ["value"],
                         gencall = make_extract_gencall(sentiment_enum)
                         )
    
    print("The above results are available in the sentiment.csv file.")
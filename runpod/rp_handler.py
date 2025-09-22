import runpod
import time
import torch
from transformers import pipeline

print("Loading model...")
pipe = pipeline("token-classification", model="abehandlerorg/econberta-ner")
print("Model loaded!")

def handler(event):
#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#       
#    Returns:
#       Any: The result to be returned to the client
    
    # Extract input data
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    # You can replace this sleep call with your own Python code
    time.sleep(seconds)  

    result = pipe(prompt)

    for dno, d in enumerate(result):
        result[dno]["score"] = float(result[dno]["score"])

    return result

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
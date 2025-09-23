import runpod
import time
from transformers import pipeline

print("Loading model...")
pipe = pipeline("token-classification", model="abehandlerorg/econberta-ner")
print("Model loaded!")

def collapse_spans(tokens):
    spans = []
    current = None

    for token in tokens:
        tag = token['entity'].split('-')[-1]
        is_continuation = current and current['tag'] == tag and token['entity'].startswith('I')

        if is_continuation:
            current['words'].append(token['word'])
            current['scores'].append(float(token['score']))
            current['end'] = token['end']
        else:
            if current:
                spans.append({
                    'tag': current['tag'],
                    'text': ''.join(current['words']).replace('▁', ' ').strip(),
                    'score': sum(current['scores']) / len(current['scores']),
                    'start': current['start'],
                    'end': current['end']
                })
            current = {
                'tag': tag,
                'words': [token['word']],
                'scores': [float(token['score'])],
                'start': token['start'],
                'end': token['end']
            }

    if current:
        spans.append({
            'tag': current['tag'],
            'text': ''.join(current['words']).replace('▁', ' ').strip(),
            'score': sum(current['scores']) / len(current['scores']),
            'start': current['start'],
            'end': current['end']
        })

    return spans


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

    result = collapse_spans(result)

    return result

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
import pandas as pd
import requests
from sklearn.preprocessing import MultiLabelBinarizer
def extract_code_from_text(text):
    # Define the API endpoint
    api_endpoint = "https://api.openai.com/v1/engines/davinci/completions"
    
    # Set your OpenAI API key
    api_key = "your_api_key"
    
    # Compose a prompt for preprocessing
    prompt = f"Extract code from the following text:\n\n{text}"
    
    # Define your request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Define the request data
    data = {
        "prompt": prompt,
        "max_tokens": 100,  # Adjust max_tokens as needed
    }
    
    # Send a POST request to the ChatGPT API
    response = requests.post(api_endpoint, headers=headers, json=data)
    
    # Extract and return the preprocessed text from the API response
    result = response.json()
    print( result)
    return result["choices"][0]["text"]
def main():
  try:
    df=pd.read_json("/content/datafinal.json")
    df_t=df[df["ContainsCode"]==True]
    df_f=df[df["ContainsCode"]==False]
    df_s=df[df["ContainsCode"]=='']
    df_s["CodeList"] = df_s['Text'].apply(extract_code_from_text)
    df_main=pd.concat([df_t,df_s,df_f]).sort_values(by="ID")
    df_main["CodeList"].fillna("",inplace=True)
    mlb = MultiLabelBinarizer()
    s1 = df_main["CodeList"]
    t1 = mlb.fit_transform(s1)
    submission = pd.DataFrame(t1)
    submission.to_csv("data_hack_sub.csv",index=False)
    return "success"
  except Exception as e:
        return f"An error occurred: {str(e)}"
main()

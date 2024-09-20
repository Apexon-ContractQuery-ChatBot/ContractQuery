import os
openai_api_key = os.getenv("OPENAI_API_KEY")
data_directory = './Synthetic.Data'
parsed_txt_directory = './parsed_txts'
vectorstore_dir = 'vectorstore'
image_path = './Apexon_logo.jpeg'
os.makedirs(parsed_txt_directory, exist_ok=True) 
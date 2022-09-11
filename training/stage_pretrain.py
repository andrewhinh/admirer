import model as md

# Load best pre-trained model from https://github.com/guilk/KAT
model_class = md.FiDT5
model_path = "/home/andrewhinh/Desktop/Projects/large_both_knowledge/"
model = model_class.from_pretrained(model_path)

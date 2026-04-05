import pandas as pd

print("1. Loading the internet dataset...")
df = pd.read_csv("dataRaw/datasetsA2.csv")

print("Original Columns: {df.columns.tolist()}")

print("\n2. Formatting to Hugging Face Standards...")

ingredients = "Other Ingredients" # <-- CHANGE THIS TO YOUR COLUMN NAME

# --- CRITICAL MENTOR LOGIC ---
# If your internet dataset DOES NOT have a label column for "Spiked", 
# we must write a script to create one based on your C0 Application rules!
def check_if_spiked(ingredient_string):
    if not isinstance(ingredient_string, str):
        return 0 # Skip empty rows
    
    # Convert to lowercase for easy searching
    ingredients = ingredient_string.lower()
    
    suspicious_aminos = ['l-glycine', 'glycine', 'taurine', 'l-taurine', 'creatine', 'Arginine', 'l-arginine', 'beta-alanine', 'l-beta-alanine', 'beta alanine', 
                         'l-beta alanine', 'beta-alanine', 'l-beta-alanine', 'beta alanine', 'l-beta alanine', 'l-citrulline', 'citrulline', 'l-citrulline malate', 
                         'citrulline malate','Glutamine', 'l-glutamine', 'glutamine', 'l-glutamine', 'glutamine', 'l-glutamine', 'glutamine','Leucine', 'l-leucine', 
                         'leucine', 'l-leucine', 'leucine', 'l-leucine', 'leucine','Isoleucine', 'l-isoleucine', 'isoleucine', 'l-isoleucine', 'isoleucine', 
                         'l-isoleucine', 'isoleucine','Valine', 'l-valine', 'valine', 'l-valine', 'valine', 'l-valine', 'valine']
    
    for amino in suspicious_aminos:
        if amino in ingredients:
            return 1 # 1 = Spiked!
            
    return 0 # 0 = Authentic/Clean

# Apply our custom thesis logic to create the "label" column
df['label'] = df[ingredients].apply(check_if_spiked)

# Rename the ingredient column to 'text' as required by DistilBERT
df = df.rename(columns={ingredients: "text"})

# Keep ONLY the text and label columns, drop everything else (like brand name, price)
df = df[['text', 'label']]

# Drop any empty rows
df = df.dropna()

print("\n3. Saving the perfectly formatted dataset...")
df.to_csv("dataset.csv", index=False)
print("Done! You can now run the train.py script using dataset.csv")
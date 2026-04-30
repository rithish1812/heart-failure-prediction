# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Heart failure prediction app with Streamlit deployment config"

# Add the remote repository
git branch -M main
git remote add origin https://github.com/rithish1812/heart-failure-prediction.git

# Push to GitHub
git push -u origin main

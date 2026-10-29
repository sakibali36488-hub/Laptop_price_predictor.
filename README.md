# Laptop Price Predictor

## Run
```
pip install -r requirements.txt
python train.py        # creates fresh pipe.pkl and df.pkl for YOUR scikit-learn version
streamlit run app.py
```
`app.py` also retrains automatically if the pickle files don't load,
so version-mismatch errors can't happen again.

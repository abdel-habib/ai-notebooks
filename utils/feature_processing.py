import numpy as np

def zscore_normalize_features(X,rtn_ms=False):
  """
  returns z-score normalized X by column
  Args:
    X : (numpy array (m,n)) 
  Returns
    X_norm: (numpy array (m,n)) input normalized by column
  """
  mu     = np.mean(X,axis=0)  
  sigma  = np.std(X,axis=0)
  X_norm = (X - mu)/sigma      

  if rtn_ms:
      return(X_norm, mu, sigma)
  else:
      return(X_norm)
import os
import pickle

feat_path = './../how2/features_semAvgPool'
new_feat_path = './../how2/features'

def load_images_AvgPool(feat_path, new_feat_path):
  img_names = [f for f in os.listdir(feat_path)]

  for idx in range(0,len(img_names)):
    imgOK = True
    # Load feature and apply average pool
    f_name = os.path.join(feat_path, img_names[idx])
    with open(f_name, "rb") as fname:
      try:
        x = pickle.load(fname)
      except:
        imgOK = False
      
      if imgOK:
        feats = x.pop('detection_features').reshape(x['num_detections'], -1, 1536)
        x['detection_features'] = feats.mean(1)
    
    # Save images
    if imgOK:
      f_name2 = os.path.join(new_feat_path, img_names[idx])
      with open(f_name2, 'wb') as fname:
        pickle.dump(x, fname, protocol=4, fix_imports=False)
    

load_images_AvgPool(feat_path, new_feat_path)

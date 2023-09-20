
'''
   ____   _   _                  _                       
  / ___| | | (_)   ___   _ __   | |_       _ __    _   _ 
 | |     | | | |  / _ \ | '_ \  | __|     | '_ \  | | | |
 | |___  | | | | |  __/ | | | | | |_   _  | |_) | | |_| |
  \____| |_| |_|  \___| |_| |_|  \__| (_) | .__/   \__, |
                                          |_|      |___/ 
                                          
The following lines of code show how to make requests to the API
'''

import requests

# ====================== Public image ====================== #

# Saving txt file
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg&save_txt=T',
                    verify=False)
print(resp.content)

# Without save txt file, just labeling the image
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg',
                    verify=False)
print(resp.content)

# You can also copy and paste the following url in your browser
'http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg'


# ====================== Public video ====================== #
# (Youtube or any public server). It is not ready (yet) to return all frames labeled while using save_txt=T. So, don't try it!

resp = requests.get('http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI',
                    verify=False)

# You can also copy and paste the following url in your browser
'http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI'


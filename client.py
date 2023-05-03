
"""
   ____   _   _                  _                       
  / ___| | | (_)   ___   _ __   | |_       _ __    _   _ 
 | |     | | | |  / _ \ | '_ \  | __|     | '_ \  | | | |
 | |___  | | | | |  __/ | | | | | |_   _  | |_) | | |_| |
  \____| |_| |_|  \___| |_| |_|  \__| (_) | .__/   \__, |
                                          |_|      |___/ 
                                          
The following lines of code show how to make requests to the API
"""



import requests


# ====================== Public image ====================== #

# Saving txt file
resp = requests.get("http://0.0.0.0:5000/detect?url=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg&save_txt=T",
                    verify=False)
print(resp.content)

# Without save txt file, just labeling the image
resp = requests.get("http://0.0.0.0:5000/detect?url=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg",
                    verify=False)
print(resp.content)

# You can also copy and paste the following url in your browser
"http://0.0.0.0:5000/detect?url=https://atlassafetysolutions.com/wp/wp-content/uploads/2019/06/ppe.jpeg"


# ====================== Public video ====================== #
# (Youtube or any public server). It is not ready (yet) to return all frames labeled while using save_txt=T. So, don't try it!

resp = requests.get("http://0.0.0.0:5000/detect?url=https://www.youtube.com/watch?v=5Mts9GGv3gk&ab_channel=CityofBoston",
                    verify=False)

# You can also copy and paste the following url in your browser
"http://0.0.0.0:5000/detect?url=https://www.youtube.com/watch?v=5Mts9GGv3gk&ab_channel=CityofBoston"


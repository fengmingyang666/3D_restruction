import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

xpath_download = "/html/body/article/div/div[2]/div/div/section/div/div/div/div[5]/div[2]/div/div[2]/div/button/span"

chrome_options = Options()
# 自定义浏览器ua
chrome_options.add_argument('user-agent="selenium user-agent"')
#chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9999")
driver = webdriver.Chrome(options=chrome_options)

NUM = 100
login_url = "https://sketchfab.com/login"
#login_url = "https://sketchfab.com/feed"
driver.get(login_url)
time.sleep(1)


# Find the username and password input fields and enter your credentials
username_input = driver.find_element("name","email")
password_input = driver.find_element("name","password")
username_input.send_keys("dcharlson644@gmail.com")
password_input.send_keys("sketchfabcom")

driver.find_element("xpath", "/html/body/main/div/div[2]/div[1]/form/div/div[2]/div[3]/div/button/span").click()
# Wait for the page to load
driver.implicitly_wait(10)

time.sleep(10)

# TODO choose the right url!!!!!!!!
#url = "https://sketchfab.com/3d-models/categories/electronics-gadgets"
url = "https://sketchfab.com/search?q=smartphone&type=models"
driver.get(url)
time.sleep(2)

#driver.find_element("xpath", "/html/body/div[3]/aside/div[1]/div[2]/div[2]/div/div[1]/div[4]/div/div/div[1]/div").click()

last_height = driver.execute_script("return document.body.scrollHeight")

load_more_count = 0
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        # hit button to load more models
        try:
            if load_more_count < 2:
                load_more_count += 1
            if load_more_count == 2:
                break  
            driver.find_element("xpath","/html/body/div[3]/main/div[2]/div/div/div/div[2]/button").click()
        except:
            break
    last_height = new_height


html_content = driver.page_source

soup = BeautifulSoup(html_content, "html.parser")
time.sleep(2)

cnt = 0
download_classes = soup.find_all("a", class_="help card-model__feature --downloads")
#print(download_classes)
for download_button in download_classes:
    if download_button:
        download_link = download_button["href"]  
        # visit the download link
        time.sleep(2)
        driver.get(download_link)
        time.sleep(2)
        # click the download button, 
        # TODO can be improved!!!!!!!!
        try:
            driver.find_element("xpath",xpath_download).click()
            #cnt = cnt + 1
        except:
            continue
        cnt = cnt + 1
    if cnt==NUM:
        break
print("All free electronic models have been downloaded.")
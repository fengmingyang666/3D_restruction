import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import os

# Configuration
# --------------------------------------------------
NUM = 100
format = ".glb"
model = "smartphone"
id = "384360712@qq.com"
pw = "sketchfabLifeng"
# --------------------------------------------------

# Set the download options
# --------------------------------------------------
options = webdriver.ChromeOptions()
# Get the current folder path
current_path = os.path.dirname(os.path.realpath(__file__))
# Set the download folder path
out_path = os.path.join(current_path, "models", model)
# check if the path exists, if not, create it
if not os.path.exists(out_path):
    os.makedirs(out_path)
prefs = {
    "profile.default_content_settings.popups": 0,
    "download.default_directory": out_path,
}
options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(options=options)
# --------------------------------------------------


# Login
# # --------------------------------------------------
# driver.get("https://sketchfab.com/login")
# time.sleep(5)

# # Accept cookies
# try:
#     accept_cookies_button = driver.find_element(
#         "xpath", "/html/body/div[5]/div[2]/div/div[1]/div/div[2]/div/button[3]"
#     )
#     actions = ActionChains(driver)
#     actions.move_to_element(accept_cookies_button).click().perform()
#     time.sleep(1)
# except:
#     pass

# # Find the username and password input fields and enter your credentials
# username_input = driver.find_element("name", "email")
# password_input = driver.find_element("name", "password")
# username_input.send_keys(id)
# password_input.send_keys(pw)
# password_input.send_keys(Keys.RETURN)
# driver.implicitly_wait(10)
# time.sleep(2)
# --------------------------------------------------

# Search
# --------------------------------------------------
url = "https://sketchfab.com/search?q=" + model + "&type=models"
driver.get(url)
time.sleep(2)
# Parse the html content and find the download button
html_content = driver.page_source
soup = BeautifulSoup(html_content, "html.parser")
download_classes = soup.find_all("a", class_="help card-model__feature --downloads")
# --------------------------------------------------

# Download
# --------------------------------------------------
cnt = 0
for download_button in download_classes:
    if download_button:
        download_link = download_button["href"]

        # visit the download link
        time.sleep(2)
        driver.get(download_link)
        time.sleep(2)

        # Parse the html content and find the download button
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        glb_div = soup.find("div", text=format)
        button = glb_div.find_parent().find_parent().find("button")
        if button is None:
            continue
        xpath = ""
        while button.parent:
            siblings = button.parent.find_all(button.name, recursive=False)
            index = siblings.index(button) + 1
            if index == 1:
                xpath = f"/{button.name}" + xpath
            else:
                xpath = f"/{button.name}[{index}]" + xpath
            button = button.parent

        # click the download button
        button = driver.find_element("xpath", xpath)
        actions = ActionChains(driver)
        actions.move_to_element(button).click().perform()
        time.sleep(2)
        cnt = cnt + 1
    if cnt == NUM:
        break
# --------------------------------------------------

# Close the browser
driver.close()
print("All free electronic models have been downloaded.")

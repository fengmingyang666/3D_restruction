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
# --------------------------------------------------
driver.get("https://sketchfab.com/login")
time.sleep(5)

# Accept cookies
try:
    accept_cookies_button = driver.find_element(
        "xpath", "/html/body/div[5]/div[2]/div/div[1]/div/div[2]/div/button[3]"
    )
    actions = ActionChains(driver)
    actions.move_to_element(accept_cookies_button).click().perform()
    time.sleep(1)
except:
    pass

# Find the username and password input fields and enter your credentials
username_input = driver.find_element("name", "email")
password_input = driver.find_element("name", "password")
username_input.send_keys(id)
password_input.send_keys(pw)
password_input.send_keys(Keys.RETURN)
driver.implicitly_wait(10)
time.sleep(2)
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
        # Find the image link, which is a .jpeg file

        img_link = download_button.find_parent().find_parent().find("img")['src']
        
        # Download the image directly and save it to the folder
        img = requests.get(img_link)
        img_name = str(cnt) + ".jpeg"
        img_path = os.path.join(out_path, img_name)
        with open(img_path, "wb") as f:
            f.write(img.content)
            
        # Find the download link
        download_link = download_button["href"]
        # visit the download link
        time.sleep(2)
        driver.get(download_link)
        time.sleep(2)

        # Parse the html content and find the download button
        html_content_inside = driver.page_source
        soup_inside = BeautifulSoup(html_content_inside, "html.parser")
        glb_div = soup_inside.find("div", text=format)
        if glb_div is None:
            # Login
            # --------------------------------------------------
            # Accept cookies
            try:
                accept_cookies_button = driver.find_element(
                    "xpath", "/html/body/div[5]/div[2]/div/div[1]/div/div[2]/div/button[3]"
                )
                actions = ActionChains(driver)
                actions.move_to_element(accept_cookies_button).click().perform()
                time.sleep(1)
            except:
                pass

            # Find the username and password input fields and enter your credentials
            try:
                username_input = driver.find_element("name", "email")
                password_input = driver.find_element("name", "password")
                username_input.send_keys(id)
                password_input.send_keys(pw)
                password_input.send_keys(Keys.RETURN)
                driver.implicitly_wait(10)
                time.sleep(2)
            except:
                continue # no preferred format
            # --------------------------------------------------
        button = glb_div.find_parent().find_parent().find("button")
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
        # Change the file name to cnt.glb
        
        # Wait for the download to complete
        while True:
            orig_filename = max([out_path + "\\" +f for f in os.listdir(out_path)],key=os.path.getctime)
            if orig_filename.endswith(format):
                break
            else:
                time.sleep(1)
        # Rename the file
        new_file_name = str(cnt) + format
        file_path = os.path.join(out_path, new_file_name)
        os.rename(orig_filename, file_path)
        cnt = cnt + 1
        
    if cnt == NUM:
        break
# --------------------------------------------------

# Close the browser
driver.close()
print("All free electronic models have been downloaded.")

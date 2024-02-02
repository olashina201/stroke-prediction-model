# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.by import By
# import time

# driver = webdriver.Chrome()
# driver.get("https://twitter.com/login")

# username = "code_advocate"
# password = "Jiggy201&"

# driver.find_element(By.CSS_SELECTOR, 'input[name="session[username]"]').send_keys(username)
# driver.find_element(By.CSS_SELECTOR, 'input[name="session[password]"]').send_keys(password)

# //*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[2]/label/div/div[2]/div/input
# /html/body/div/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[2]/label/div/div[2]/div/input
# driver.find_element(By.XPATH, "//*[@id='layers']/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[2]/label/div/div[2]/div/input[@name='username']").send_keys(username)

# driver.find_element(By.CSS_SELECTOR, 'div[data-testid="LoginForm_Login_Button"]').click()

# Find the 'Next' button using its XPATH and click it to move to the password field
# driver.find_element(By.XPATH, "//span[contains(text(),'Next')]").click()

# Wait for the next page to load before continuing
# time.sleep(3)

# Find the password input field using its XPATH and enter a password
# driver.find_element(By.XPATH, "//input[@name='password']").send_keys(password)

# Find the 'Log in' button using its XPATH and click it to log in
# log_in = driver.find_element(By.XPATH,"//span[contains(text(),'Log in')]")
# log_in.click()

# time.sleep(15)

# user_profile = "https://twitter.com/code_advocate"
# driver.get(user_profile)

# Scroll down to load more tweets (you might need to adjust the loop and sleep duration)
# for _ in range(3):
#     driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
#     time.sleep(2)

# Extract tweets
# tweets = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweet"]')
# tweets = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")
# print('TWEETS', tweets);
# for tweet in tweets:
#     print(tweet)

# Close the webdriver
# driver.quit()



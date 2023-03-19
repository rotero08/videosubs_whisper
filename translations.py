import os
import time
from selenium import webdriver
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchWindowException
from selenium.common.exceptions import TimeoutException

def translate(output_filename):
    while True:
        try:
            os.system("taskkill /f /im tor.exe")

            # Start the Tor process
            torexe = os.popen(r'C:/Users/Rafael Otero/Desktop/Tor Browser/Browser/TorBrowser/Tor/tor.exe')

            # Wait for the Tor process to start
            time.sleep(10)

            # Configure the Firefox profile to use Tor
            profile = FirefoxProfile(r'C:/Users/Rafael Otero/Desktop/Tor Browser/Browser/TorBrowser/Data/Browser/profile.default')
            profile.set_preference('network.proxy.type', 1)
            profile.set_preference('network.proxy.socks', '127.0.0.1')
            profile.set_preference('network.proxy.socks_port', 9050)
            profile.set_preference("network.proxy.socks_remote_dns", False)
            #profile.set_preference('intl.accept_languages', 'en-US, en')
            profile.set_preference("browser.download.folderList", 2)
            profile.set_preference("browser.download.manager.showWhenStarting", False)
            profile.set_preference("browser.download.dir", output_filename[:output_filename.rfind('/')])
            #profile.set_preference("browser.download.dir", output_filename[:output_filename.rfind('/')])
            profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            profile.update_preferences()

            # Launch Firefox using the Tor profile
            firefox_options = webdriver.FirefoxOptions()
            firefox_options.binary_location = r'C:/Program Files/Mozilla Firefox/firefox.exe'
            driver = webdriver.Firefox(firefox_profile=profile, options=firefox_options, executable_path=r'C:/WebDrivers/geckodriver.exe')

            driver.get("https://www.deepl.com/es/translator/files")
            from selenium.webdriver.support.wait import WebDriverWait
                

            # Handle cookie banner if it appears
            try:
                cookie_banner = WebDriverWait(driver, timeout=15).until(lambda d: d.find_element(By.XPATH, "//button[@dl-test='cookie-banner-strict-accept-selected']"))
                cookie_banner.click()
            except NoSuchWindowException:
                pass

            upload_button = WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.XPATH, "//input[@dl-test='doctrans-upload-input']"))
            upload_button.send_keys(output_filename.replace("/","\\"))
                
            # Find the target language dropdown and click it
            target_lang_dropdown = WebDriverWait(driver, timeout=15).until(lambda d: d.find_element(By.XPATH, "//button[@data-testid='doctrans-target-lang-btn']"))
            target_lang_dropdown.click()

            # Select the Spanish language
            lang_selection = WebDriverWait(driver, timeout=15).until(lambda d: d.find_element(By.XPATH, "//button[@dl-test='document-translator-lang-option-es']"))
            lang_selection.click()

            # Find the download button and click it
            translate_button = WebDriverWait(driver, timeout=15).until(lambda d: d.find_element(By.XPATH, "//button[@dl-test='doctrans-translation-button']"))
            translate_button.click()

            output_filename = output_filename.replace(".docx" , " es.docx")
            time.sleep(180)
            break # If the element is found, break out of the loop
        except TimeoutException:
            driver.quit() # If the element is not found, refresh the page and try again

        except Exception as e:
            driver.quit()

    return output_filename
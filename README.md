Phishing Webpage Detector with Html features trained by linear support vector machine

This project helps people prevent phishing attacks light weightedly. It is based on the research paper: "DeltaPhish: Detecting Phishing Webpages in Compromised Websites" by Igino, et al. Datasets used are also from the authors.

It takes in Html code of a webpage, along with html code of its corresponding homepage, generating comparison based features based primarily on jaccard index. These features are generated from:

- Url set (href attribute)
- 2LD set (second level domains of the Url set)
- SS (styles tag)
- SS-url (links to external stylesheets)
- SS-2LD
- I-url (links to images)
- I-2LD
- Title
- Language
- Xlink (a binary variable denoting whether two pages are linked)

Getting Started
On git bash/Windows/Mac/Linux cmd:
git clone https://github.com/EdwinChenXM/PhishingWebpageDetector

Prerequisites
You do not need anything since the interpreter is contained in the virtual environment, and all libraries are installed within, If you have any troubles locating the python interpreter, it is at venv/bin/python. The project is written in python 2.7

How to use it:

To start with, you're going to need to crawl HTML code from webpages yourself since the project does not contain that functionality just yet. A useful link: https://stackoverflow.com/questions/3533528/python-web-crawlers-and-getting-html-source-code
You need to index the HTML files in a way, then create metadata about it in json. 
Example of that is in: 
- HTML codes: HTML/xxx
- Metadata: deltaphish_data.json

There's generally five steps:
1. Extract the features from raw html code, store the features in .csv file
2. Extract numerical values from the stored features, store the feature vectors in .csv file
3. Split the data into a training set and a test set
4. Train a model based on these feature vectors
5. Test the model 

The project is logically distributed into 3 python scripts:
html_features, feature_engineering, svm_training

Optional Functionality:
You can plot scatter plots of all data points

1. Extracting features from html code:
you can do so by using the function html_features.extractHtmlFeatures(filename1, filename2, filename3)
extracted features would then be stored in the 3 filenames provided.
  1: all of the feature vectors
  2: feature vectors of phishing webpages
  3: feature vectors of legitimate webpages
Do note that filenames are default parameters, if you don't plan to change them, everything remains functional if you don't input any filenames, however if you do, make sure you name them and use them consistently throughout

html_features.parseData(filename1, filename2, filename3) is a helper method for you to reload the stored data into python, it returns them in formats of pandas dataframe

2. Extracting numerical values from stored features/Split data into training set and test set:
you can do so by using the function svm_training.generateTrainTestData()


until finished
End with an example of getting some data out of the system or using it for a little demo

Running the tests
Explain how to run the automated tests for this system

Break down into end to end tests
Explain what these tests test and why

Give an example
And coding style tests
Explain what these tests test and why

Give an example
Deployment
Add additional notes about how to deploy this on a live system

Built With
Dropwizard - The web framework used
Maven - Dependency Management
ROME - Used to generate RSS Feeds
Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

Versioning
We use SemVer for versioning. For the versions available, see the tags on this repository.

Authors
Billie Thompson - Initial work - PurpleBooth
See also the list of contributors who participated in this project.

License
This project is licensed under the MIT License - see the LICENSE.md file for details

Acknowledgments
Hat tip to anyone whose code was used
Inspiration
etc

# Sahibinden.com #dev akademi

I have done 2 different projects. However, second one was not tested with the given data.
## First Project
According to person features, it predicts the possible ad category that he/she could click. (Random Forest Algorithm)
To use this project first, run "prepare_dataset.py" file to prepare the data with features("education", "job", "marital_status",
            "birt_year", "gender", "user_city") and labels('İkinci El ve Sıfır Alışveriş', 'Emlak', 'Vasıta', None
       'İş Makineleri & Sanayi','Yedek Parça, Aksesuar, Donanım & Tuning', 'Hayvanlar Alemi')
  
Before run that, please change the line 10 in "prepare_dataset.py" file with  "file = codecs.open("../all_data 3.json", "r", "utf-8")" this. It creates a Dataset folder with .npy files. I have filtered the ads that has clicked more than once from a unique user.
After runing that, you can run "clickable_category_prediction_for_user.py" file to see the results.
I have used K-fold cross validation.Then, I have printed out the accuracies for each 10 itterations. 
Results:~%72 percent accuracy for 7 different labels.

## Second Project
The prediction of the title whether it is clickable or not . It evaluates the title strength whether it will get the 'Click' or not by using the Naive Bayes Algorithm. But unfortunately, data were encoded,and words were mixed-up with the punctution. Furthermore, the turkish words generally have the additional letters from the end of the word.Consequently, it makes words different such as; "ev" and 
"ev-i". Thus, it was hard to determine ".,!,etc" characters. But it works in real sentences.


# Prerequires:
- numpy
- sklearn
- codecs
- json
- spacy 

PS: Thanks for everything. It was an honour to be a part of such a wonderful event:)


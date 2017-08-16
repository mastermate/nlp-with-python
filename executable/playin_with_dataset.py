import os
import codecs

data_dir = os.path.join('..','data','yelp_dataset_challenge_academic_dataset')
business_fpath = os.path.join(data_dir,'yelp_academic_dataset_business.json')

with codecs.open(business_fpath, encoding = 'utf_8') as f:
	business_entry = f.readline()

print(business_entry)
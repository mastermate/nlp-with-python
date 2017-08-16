import os
import codecs
import json

data_dir = os.path.join('..','data','yelp_dataset_challenge_academic_dataset')

def main():
	retrieveAndStoreReviews('reviews.txt')

def retrieveRestaurantsIds():
	business_fpath = os.path.join(data_dir,'yelp_academic_dataset_business.json')
	restaurant_ids = set()
	with codecs.open(business_fpath, encoding = 'utf_8') as f:
		for business_json in f:
			business = json.loads(business_json)
			#print(business)
			cats = business[u'categories']
			if (cats != None and u'Restaurants'  in cats):
    			# add the restaurant business id to our restaurant_ids set
				restaurant_ids.add(business[u'business_id'])

	# turn restaurant_ids into a frozenset, as we don't need to change it anymore
	restaurant_ids = frozenset(restaurant_ids)
	
	return restaurant_ids
	

def retrieveAndStoreReviews(outputName):
	restaurant_ids = retrieveRestaurantsIds()
	
	reviews_fpath = os.path.join(data_dir,'yelp_academic_dataset_review.json')
	intermediate_directory = os.path.join('..', 'intermediate')
	review_txt_filepath = os.path.join(intermediate_directory, outputName)
	
	counter = 0
	with codecs.open(reviews_fpath, encoding = 'utf_8') as fjson, codecs.open(review_txt_filepath, 'w',encoding = 'utf_8') as freviews:
		for review_json in fjson:
			review = json.loads(review_json)
			if (review[u'business_id']  in restaurant_ids):
				text = review[u'text']
				if (text != None):
					counter += 1
					freviews.write(text.replace('\n','\\n') + '\n')	
	print(counter,' total reviews added to ', outputName, ' file.')			
				

if __name__ == '__main__':
	main()
				
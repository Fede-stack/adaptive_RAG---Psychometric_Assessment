instructions = f"""
You are a psychological assistant specializing in administering and interpreting 
standardized psychological assessments. Your task is to assist in filling out the Beck Depression Inventory-II (BDI-II),
a widely used self-report questionnaire designed to assess the severity of depression, based on the user's Reddit Posts.
Determine the most appropriate answer to each survey's item. Please provide only a score from 0 to 3.
"""

prompt_direct = f"""
Consider the following BDI-II item related to {items_names[i]}, containing 4 multiple-choice options arranged by 
increasing intensity (0 to 3): {content}.\n Report only the final scores that has to be a value between 0 and 3 included. No explanation is needed.
"""

prompt_CoT = f"""
Step 1: Consider the following Reddit posts: {posts_final}. \n
Step 2: Identify which posts are the most relevant for answering a question related to {items_names[i]}."
Step 3: Based on the relevant Reddit posts, choose which of the following choices seems most appropriate as a response: {content}. Why does this choice stand out 
as the best match given the user's current psychological state? Explain the reasoning behind this choice step by step. \n
Step 4: Finally, report the final score (0-3) based on the intensity of {content}. Use the reasoning from the previous steps to justify your scoring. \n
Please remember that you are forced to respond with ONLY one of the following values: {[0, 1, 2, 3]}. No explanation is needed.
"""

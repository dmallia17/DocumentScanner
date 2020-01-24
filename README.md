# DocumentScanner
## TTP Capstone Project

### Author: Daniel Mallia and Sajarin Dider
### Date Begun: 1/17/2020

## Run Instructions:
- Set up environment using pip/conda with requirements.txt
- Run with "python3 scan.py [IMAGENAME.jpg]" or, given permission, "./scan.py [IMAGENAME.jpg]"

## To-Do List:
- [x] Complete transformForOCR function
- [x] Begin work on selecting individual characters
- [x] Incorporate EAST detector.
- [x] Begin work on organizing characters for recognition
- [x] Begin work on data preprocessing for network training
- [ ] Refine network accuracy: most likely via data augmentation.
- [x] Pip freeze for requirements.txt - update to include Pillow
- [x] Update notice regarding code used from article in char.py
- [x] Create loop for processing characters.
- [ ] Add in checks to remove bad character localizations
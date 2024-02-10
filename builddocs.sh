jupyter nbconvert --to markdown --output=readme examples/from_text_to_object/from_text_to_object.ipynb

jupyter nbconvert --to markdown --output=readme examples/quick_meeting/quick_meeting.ipynb

jupyter nbconvert --to markdown --output=readme examples/tough_meeting/tough_meeting.ipynb
jupyter nbconvert --to markdown --output=readme_dictype examples/tough_meeting/tough_meeting_dictype.ipynb

jupyter nbconvert --to markdown --output=readme examples/extract/extract.ipynb
jupyter nbconvert --to markdown --output=readme_dictype examples/extract/extract_dictype.ipynb

jupyter nbconvert --to markdown --output=readme examples/tag/tag.ipynb

jupyter nbconvert --to markdown --output=readme examples/compare/compare.ipynb

jupyter nbconvert --to markdown --output=readme examples/interact/interact.ipynb

# mkdocs build
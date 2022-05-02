# Utilities for NER with Huggingface
Contains:

- Tagging utilities for PyLighter
- Active-learning
- Utils to work with Huggingface predictions

# TO-DO, checks, fixes...
- https://gab41.lab41.org/lessons-learned-fine-tuning-bert-for-named-entity-recognition-4022a53c0d90

# Current known bugs/issues
- `[PAD]` tokens are being classified
- Sentencepiece tokens that are not starters (e.g. `'#ation'`) are tagged as `I-`, but they're disregarded in the loss fn (by setting the label `-100`). However, these tags still appear in predictions (Why?)

# Possible refactor
Since this project is one year(?) old, fast-growing huggingface library is releasing new ways of doing things, new tutorials, etc. So it would be a good idea to refactor this library.
Also, instead of using PyLighter, it would be great to support a better, collaboratively and standardized annotation tool like Doccano.

- https://huggingface.co/docs/transformers/tasks/token_classification


- https://www.freecodecamp.org/news/getting-started-with-ner-models-using-huggingface/

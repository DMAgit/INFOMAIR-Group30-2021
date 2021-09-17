rule_dict = {
    "kay": "ack",
    "okay": "ack",
    "correct": "affirm",
    "right": "affirm",
    "ye": "affirm",
    "yea": "affirm",
    "yeah": "affirm",
    "yes": "affirm",
    "bye": "bye",
    "goodbye": "bye",
    "does": "confirm",
    "is": "confirm",
    "dont": "deny",
    "wrong": "deny",
    "hello": "hello",
    "hi": "hello",
    "no": "negate",
    "and": "null",
    "breathing": "null",
    "cough": "null",
    "inaudible": "null",
    "knocking": "null",
    "noise": "null",
    "music": "null",
    "sil": "null",
    "um": "null",
    "unintelligible": "null",
    "back": "repeat",
    "repeat": "repeat",
    "another": "reqalts",
    "else": "reqalts",
    "anything": "reqalts",
    "about": "reqalts",
    "address": "request",
    "phone": "request",
    "what": "request",
    "whats": "request",
    "post": "request",
    "telephone": "request",
    "area": "request",
    "can": "request",
    "give": "request",
    "could": "request",
    "location": "request",
    "type": "request",
    "start": "restart",
    "over": "restart",
    "thank": "thankyou",
    "great": "thankyou"
}

count_dict = {
    "ack": 0,  # acknowledgment
    "affirm": 0,  # positive confirmation
    "bye": 0,  # greeting at the end of the dialog
    "confirm": 0,  # check if given information confirms to query
    "deny": 0,  # reject system suggestion
    "hello": 0,  # greeting at the start of the dialog
    "inform": 0,  # state a preference or other information
    "negate": 0,  # negation
    "null": 0,  # noise or utterance without content
    "repeat": 0,  # ask for repetition
    "reqalts": 0,  # request alternative suggestions
    "reqmore": 0,  # request more suggestions
    "request": 0,  # ask for information
    "restart": 0,  # attempt to restart the dialog
    "thankyou": 0,  # express thanks
}

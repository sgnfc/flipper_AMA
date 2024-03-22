from langchain.evaluation.qa import QAGenerateChain

from langchain.evaluation.qa import QAEvalChain

from langchain.chat_models import ChatOpenAI


pinball_examples = [
    {"query": "In Medival Madness, Do you get an extra ball from shooting the castle twice?",
    "response": "Yes"
    },
    {"query": "In Addams family, what will you get if you hit the ball into Thing's eject?",
     "response": "Skill shot"}

]

def generate_examples(llm, docs):


    #chain.add_examples(pinball_examples)
    example_gen_chain = QAGenerateChain.from_llm(llm)
    new_examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in docs[:5]]
    )
    print("New examples:", new_examples)
    return new_examples


""" Use an LLM to compare your model predictions with the generated examples of query <> repsonse pairs."""
def llm_assisted_evaluation(examples, predictions):
    llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)

    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['text'])
        print()
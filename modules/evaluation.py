import json
import os
from langchain.evaluation.qa import QAGenerateChain

from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


pinball_examples = [
    {"query": "In Medival Madness, Do you get an extra ball from shooting the castle twice?",
    "response": "Yes"
    },
    {"query": "In Addams family, what will you get if you hit the ball into Thing's eject?",
     "response": "Skill shot"}

]

def generate_examples(llm, docs):

    parser = JsonOutputFunctionsParser()
    #chain.add_examples(pinball_examples)
    example_gen_chain = QAGenerateChain.from_llm(llm)
    new_samples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in docs[:5]]
    )
    for i in range(len(new_samples)):
        new_samples[i] = new_samples[i]['qa_pairs']
    print("New examples:", new_samples)
    return new_samples


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

def evaluate_agent(agent, docs, use_previous_samples=False):
    automatic_examples_fp = 'data/pinball_examples.json'
    hard_coded_examples_fp = 'data/manual_pinball_examples.json'
    llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    if not use_previous_samples:
        print("Generating new examples")
        automatic_examples = generate_examples(llm, docs)
    else:
        print("Using previous examples")
        if os.path.exists(automatic_examples_fp):
            automatic_examples = json.load(open(automatic_examples_fp))
        else:
            print("No previous examples found")
            return

    hard_coded_samples = json.load(open(automatic_examples_fp)) if os.path.exists(hard_coded_examples_fp) else None

    
    predictions = automatic_examples + hard_coded_samples
    print("Done generating examples, invoking agent...")
    for idx,sample in enumerate(predictions):
        query = sample['query']
        result = agent.invoke({"query":query})['output']
        predictions[idx]['result'] = result
    
    print("Done invoking agent, evaluating...")


    
    evaluator = load_evaluator("pairwise_embedding_distance")

    total_scores = []
    for idx,sample in enumerate(predictions):
        answer = sample['answer']
        result = sample['result']
        score = evaluator.evaluate_string_pairs(prediction=answer, prediction_b=result)
        total_scores.append(score)
    scores = [d['score'] for d in total_scores]
    average_score = sum(scores)/len(scores)
    return scores, average_score
    
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


class QuestionAnswerer:
    def __init__(self, model_path="question_answerer/psr_qa_fine_tuned_model"):
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    def query_context(self, question, context):
        # Tokenize the input
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)

        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the answer (start and end token positions)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        answer.replace(",", "")  # Commas will ruin the csv file
        return answer


if __name__ == "__main__":
    question_answerer = QuestionAnswerer(model_path="psr_qa_fine_tuned_model")
    address_examples = [
        "ct. 1: commercial  burglary  ct. 2: theft of  property/  pulaski county circuit  court, little rock, ar;  docket no.: cr 19- 4529  t was represented by counse urglarizing the office and pro as, on or about september 2 hours and stole items.",
        "illegal re-entry into the  united states after  deportation/  united states district  court, western district  of texas, del rio  division, tx;  docket no.: dr-10- cr-0066-001am  nt was represented by cou ong with six illegal aliens,  iles east of del rio, texa was detained after a short fo ip and he admitted he was a als, did not have any docum s legally. the defendant, al border patrol station for fur eak with agents without the  nt, along with a group of  r, via a boat, near the del r gh the brush and waited fo mputerized records check r ons by border patrol agents",
        "ct. 1: 647(b) pc,  disorderly conduct  (misd.).  los angeles, ca, pd",
        "496(a) pc: receiving  stolen property  (fel.)/ los angeles  superior court;  8",
        "37-23732b(a)(2):  drug trafficking in  cocaine (cts. 2 - 3)/  canyon county  district court, ut;  docket no.: cr- 2009-20518  represented by counsel.  o court records an addition ed."
    ]
    for example in address_examples:
        answer = question_answerer.query_context("what is the courthouse address?", example)
        print(f"Answer: {answer}")

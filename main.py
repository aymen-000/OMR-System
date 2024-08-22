import cv2 as cv 
import numpy as np 
import preprocessing
import postProcessing
import argparse

###########################################
# Default values
width = 700
height = 700
choices = 5
questions = 5
##########################################

def main(img_path, width, height, questions, choices, right_answers, save_file): 
    img, imgGray, imgBlur, imgCanny, contours = preprocessing.preprocessing(img_path, width, height)
    
    imgWrapped, gradWrapped, invMatrix, invGradePerspectiveMertix = preprocessing.find_recatngles(
        contours=contours, img=img, width=width, height=height
    )
    
    score, score_for_each, ans = preprocessing.get_final_score(
        imgWrapped=imgWrapped, questions=questions, choices=choices  ,
        right_answers=right_answers
    )
    
    imgFinal = postProcessing.get_final_results(
        imgWarpped=imgWrapped, 
        score=score, 
        img=img , 
        ans = ans , 
        width=width , 
        height=height , 
        file_to_save=save_file, 
        right_answers=right_answers, 
        score_for_each=score_for_each, 
        choices=choices, 
        questions=questions,
        gradeWarpped=gradWrapped, 
        invMatrix=invMatrix,
        invGradePerspectiveMertix=invGradePerspectiveMertix
    )
    return score 
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="OMR System")
    
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--width", type=int, default=700, help="Width of the output image")
    parser.add_argument("--height", type=int, default=700, help="Height of the output image")
    parser.add_argument("--questions", type=int, default=5, help="Number of questions")
    parser.add_argument("--choices", type=int, default=5, help="Number of choices per question")
    parser.add_argument("--right_answers", type=int, nargs='+', required=True, help="List of correct answers (e.g., 1 2 0 0 4)")
    parser.add_argument("--save_file", type=str, required=True, help="File path to save the final output image")

    args = parser.parse_args()

    score = main(
        img_path=args.img_path,
        width=args.width,
        height=args.height,
        questions=args.questions,
        choices=args.choices,
        right_answers=args.right_answers,
        save_file=args.save_file
    )
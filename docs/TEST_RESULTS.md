# Test Results

Generated on 2026-11-22 by `python -m scripts.evaluate` (full RAG with gemini-3.5-flash-lite).
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, top_k=4, min_similarity=0.25.

| # | Topic | Question | Expected | Source returned | Answer |
|---|-------|----------|----------|-----------------|--------|
| 1 | Fees | What are the total fees for the bootcamp? | Page 16 | Page 16 | The total fees for the bootcamp are R 38,950. |
| 2 | Key dates | When is orientation day? | Page 7 | Page 7 | Orientation Day is scheduled for 15 January 2026 at 10 am. |
| 3 | Live classes | What days and times are the live classes? | Page 11 | Page 11 | Live classes are held from 6 pm to 8 pm on Thursdays and from 9 am to 11 am on Tuesdays for the first 12 weeks of the bootcamp, after which they reduce to 1 live class every Tuesday at 9 am - 11 am. |
| 4 | Hardware | How much RAM does my laptop need? | Page 5 | Page 5 | At least 4 or 8 GB of RAM is required for efficient multitasking and handling of development environments. |
| 5 | Grading | How much is the final project worth in the final grade? | Page 14 | Page 14 | The final project is worth 35% of the final grade. |
| 6 | Tutor support | When are the tutor support hours? | Page 20 | Page 20 | Tutors host support hours every Tuesday from 2 pm to 4 pm and 6 pm to 8 pm, and on Thursdays from 10 am to 12 pm and 6 pm to 8 pm. |
| 7 | Communication | Which platform is used for communication after orientation day? | Page 18 | Page 18 | Discord is used for communication after orientation day. |
| 8 | Fees | What financing options are available to pay the fees? | Pages 16, 17 | Pages 16, 17 | The financing options available for paying the fees are through Zaio's partnered financing options with Capitec and Manati to facilitate the process of securing a study loan. |
| 9 | Calendar | What does a yellow tile on my calendar mean? | Page 9 | Page 9 | A yellow tile indicates late completion, meaning the task was completed after the designated day or more. While not ideal, it serves as a valuable learning opportunity to enhance time management skills. |
| 10 | Placement | How much do paid internships pay per month? | Page 24 | Page 24 | Paid internships provide a monthly salary of R4000. |
| 11 | Team | Who is the lead instructor? | Page 15 | Page 15 | Kago Kola is the lead instructor. |
| 12 | Discord levels | What is the highest level tag on Discord? | Page 21 | Page 21 | The highest level tag in the Code Mentor category is Lvl 30 Code Mentor Pro Max, and the highest level in the Team Player category is Lvl 18 Team Player Ultra Pro Max. |
| 13 | Sprints | Which technologies are covered in weeks 16 to 20? | Page 6 | Page 6 | In weeks 16 to 20, the technologies covered are NodeJS, NPM, Express, REST APIs, MongoDB, Postman, and Async & Await. |
| 14 | Out of scope | What is the attendance requirement? | Not in handbook | N/A | I don't know. That information is not available in the handbook. |
| 15 | Out of scope | What is the bootcamp's refund policy? | Not in handbook | N/A | I don't know. That information is not available in the handbook. |
| 16 | Out of scope | What is the capital of France? | Not in handbook | N/A | I don't know. That information is not available in the handbook. |

Pages retrieved for each question (before answer generation):

1. What are the total fees for the bootcamp?: Pages 7, 16, 17, 19
2. When is orientation day?: Pages 7, 8, 9, 18
3. What days and times are the live classes?: Pages 7, 9, 11, 12
4. How much RAM does my laptop need?: Page 5
5. How much is the final project worth in the final grade?: Pages 14, 23
6. When are the tutor support hours?: Pages 8, 11, 20
7. Which platform is used for communication after orientation day?: Pages 9, 13, 18
8. What financing options are available to pay the fees?: Pages 16, 17
9. What does a yellow tile on my calendar mean?: Pages 8, 9
10. How much do paid internships pay per month?: Pages 16, 17, 24, 25
11. Who is the lead instructor?: Pages 2, 13, 15, 20
12. What is the highest level tag on Discord?: Pages 21, 22
13. Which technologies are covered in weeks 16 to 20?: Pages 6, 8, 9
14. What is the attendance requirement?: Pages 7, 11, 12, 15
15. What is the bootcamp's refund policy?: Pages 7, 11, 16, 17
16. What is the capital of France?: none above threshold

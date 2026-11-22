# Test Results

Generated on 2026-11-22 by `python -m scripts.evaluate` (full RAG with `gemini-3.5-flash-lite`).
Embedding model `sentence-transformers/all-MiniLM-L6-v2`, top_k=6, min_similarity=0.3. Knowledge base: 40 handbook chunks, 523 website chunks.

**21/21 passed.** Handbook: 7/7, Website: 8/8, Both: 2/2, Unanswerable: 4/4.

A test passes when an answerable question is answered with one of the expected sources,
or when an unanswerable question returns the fallback message:
*"I could not find that information in the available knowledge base."*

| # | Category | Question | Retrieved source | Generated answer | Result |
|---|----------|----------|------------------|------------------|--------|
| 1 | Handbook | When are the tutor support hours? | Student Handbook - Page 20 | Tutors host support hours every Tuesday from 2 pm to 4 pm and 6 pm to 8 pm, and on Thursdays from 10 am to 12 pm and 6 pm to 8 pm. | Pass |
| 2 | Handbook | What days and times are the live classes? | Student Handbook - Page 11<br>Student Handbook - Page 12 | For the first 12 weeks of the bootcamp, live classes are held from 6 pm to 8 pm on Thursdays and 9 am to 11 am on Tuesdays. After the first 12 weeks, live classes reduce to 1 live class every Tuesday from 9 am to 11 am. | Pass |
| 3 | Handbook | What does a yellow tile on my calendar mean? | Student Handbook - Page 9 | A yellow tile indicates late completion, meaning the task was completed after the designated day or more. While not ideal, it serves as a valuable learning opportunity to enhance time management skills. | Pass |
| 4 | Handbook | Which platform is used for communication after orientation day? | Student Handbook - Page 18 | After orientation day, all ongoing communication transitions to Discord, which serves as the primary platform and central hub for discussions, announcements, and real-time interaction. | Pass |
| 5 | Handbook | How much is the final project worth in the final grade? | Student Handbook - Page 14 | The final project (React + Node) is worth 35% of the final grade. | Pass |
| 6 | Handbook | How much RAM does my laptop need for the bootcamp? | Student Handbook - Page 5 | At least 4 or 8 GB of RAM is required for efficient multitasking and handling of development environments. | Pass |
| 7 | Handbook | What is the highest level tag on the Discord server? | Student Handbook - Page 21 | The highest level tag in the Code Mentor category is Lvl 30 Code Mentor Pro Max, and the highest level in the Team Player category is Lvl 18 Team Player Ultra Pro Max. | Pass |
| 8 | Website | What courses does ZAIO offer? | https://www.zaio.io/<br>https://www.zaio.io/2027 | ZAIO offers several accredited tech and business courses and programmes, including Occupational Certificates (such as Software Developer, AI & ML Engineer, Entrepreneurship & Business Management, and Graphic Design & Media) as well as focused, job-skills bootcamps. | Pass |
| 9 | Website | What is ZAIO's refund policy? | https://www.zaio.io/refund-policy<br>https://www.zaio.io/terms | What you can get back depends on your payment plan and timing. You can get a full refund if you cancel in writing before your programme starts (or before orientation on 12-, 18-, 24-month and financed plans). On upfront, two-month and three-month plans, you can get 75% of the invoiced amount within 30 days of the start; nothing is refundable after that. If Zaio cancels your cohort, you can choose a full refund or a free transfer to the next cohort. Cancellations must be sent in writing to accounts@zaio.io. | Pass |
| 10 | Website | When does the next Fullstack AI Engineer bootcamp start? | https://www.zaio.io/programmes/fullstack-ai-engineer-bootcamp<br>https://www.zaio.io/2027 | The Fullstack AI Engineer bootcamp starts on 28 January 2027. | Pass |
| 11 | Website | Where are ZAIO's campuses located? | https://www.zaio.io/about | Zaio's campuses are located in Cape Town (66 Albert Road, Woodstock) and Durban (221 Musgrave Road, Berea). | Pass |
| 12 | Website | Is ZAIO accredited? | https://www.zaio.io/accreditation<br>https://www.zaio.io/<br>https://www.zaio.io/2027/parents | Yes, Zaio Institute of Technology is accredited and quality assured by the Quality Council for Trades and Occupations (QCTO) as a Skills Development Provider to offer Occupational Certificates registered on the National Qualifications Framework. | Pass |
| 13 | Website | What are the entry requirements for the occupational certificates? | https://www.zaio.io/2027<br>https://www.zaio.io/entry-requirements | An Occupational Certificate generally asks for matric (an NQF Level 4 qualification) or an equivalent. For some specific courses, the requirements differ: Entrepreneurship & Business Management requires Grade 10 and up (an NQF Level 2 qualification with mathematical literacy and communication or equivalent); AI & ML Engineer requires Grade 12 with English and Mathematical Literacy, or an NQF Level 4 qualification; and Data Science requires an NQF Level 4 qualification (e.g., Matric) or equivalent along with basic mathematics and logical thinking skills. If you do not have matric, Recognition of Prior Learning (work experience or earlier study) can count toward eligibility. | Pass |
| 14 | Website | How do I apply to ZAIO? | https://www.zaio.io/how-to-apply | To apply to ZAIO, complete the online application by providing your details, background, and desired programme, which takes about five minutes and requires no documents to start. Afterwards, you will talk to an admissions consultant who will call or message you to confirm the programme, answer questions, and discuss payment. Next, you choose how to pay (in full, over two months, or financed through Manati). Once the consultation is done and tuition is arranged, you will receive your confirmation by email. | Pass |
| 15 | Website | What is The Weekly Commit? | https://www.zaio.io/newsletter | The Weekly Commit is a free tech career newsletter by Zaio that lands in your inbox every Thursday, featuring practical answers, honest perspectives on tech careers, and a weekly motivational nudge. | Pass |
| 16 | Both | How much does the Fullstack AI Engineer bootcamp cost? | Student Handbook - Page 1<br>Student Handbook - Page 16<br>https://www.zaio.io/programmes/fullstack-ai-engineer-bootcamp | The Student Handbook states that the total fees for the Full-Stack AI Engineer Bootcamp are R38,950, with an upfront payment option. However, the Zaio website states that the upfront fee is R42,950, which is the lowest total price for this bootcamp, and that you can also pay from R4,178 a month through Manati or in instalments over the programme, both of which cost more in total. | Pass |
| 17 | Both | Which companies can I use to finance my studies? | Student Handbook - Page 16<br>Student Handbook - Page 17 | You can use Capitec and Manati as financing partners to secure a study loan for the bootcamp. | Pass |
| 18 | Unanswerable | What is the capital of France? | None (refused) | I could not find that information in the available knowledge base. | Pass |
| 19 | Unanswerable | Who won the 2022 FIFA World Cup? | None (refused) | I could not find that information in the available knowledge base. | Pass |
| 20 | Unanswerable | What is the Wi-Fi password at the Cape Town campus? | None (refused) | I could not find that information in the available knowledge base. | Pass |
| 21 | Unanswerable | What will the weather be in Cape Town tomorrow? | None (refused) | I could not find that information in the available knowledge base. | Pass |

## Chunks retrieved before generation

Sources of the chunks passed to the model as context, most similar first.

1. **When are the tutor support hours?**: Student Handbook - Page 20, Student Handbook - Page 11, https://www.zaio.io/faculty, Student Handbook - Page 8, https://www.zaio.io/programmes/fullstack-web-development-bootcamp
2. **What days and times are the live classes?**: Student Handbook - Page 11, https://www.zaio.io/intakes, Student Handbook - Page 12, https://www.zaio.io/events/class-of-2026-graduation, https://www.zaio.io/faculty, Student Handbook - Page 7
3. **What does a yellow tile on my calendar mean?**: Student Handbook - Page 9
4. **Which platform is used for communication after orientation day?**: Student Handbook - Page 18, Student Handbook - Page 9
5. **How much is the final project worth in the final grade?**: Student Handbook - Page 14, https://www.zaio.io/programmes/software-developer-occupational-certificate, https://www.zaio.io/events/class-of-2026-graduation, https://www.zaio.io/programmes/graphic-media-designer-occupational-certificate, Student Handbook - Page 23
6. **How much RAM does my laptop need for the bootcamp?**: Student Handbook - Page 5, Student Handbook - Page 1, https://www.zaio.io/programmes/fullstack-web-development-bootcamp, https://www.zaio.io/entry-requirements, https://www.zaio.io/programmes/cybersecurity-bootcamp
7. **What is the highest level tag on the Discord server?**: Student Handbook - Page 22, Student Handbook - Page 21, Student Handbook - Page 18, https://www.zaio.io/community
8. **What courses does ZAIO offer?**: https://www.zaio.io/, https://www.zaio.io/2027, https://www.zaio.io/for-corporate, Student Handbook - Page 4, Student Handbook - Page 20
9. **What is ZAIO's refund policy?**: https://www.zaio.io/refund-policy, https://www.zaio.io/terms, Student Handbook - Page 26, Student Handbook - Page 17
10. **When does the next Fullstack AI Engineer bootcamp start?**: https://www.zaio.io/programmes/fullstack-ai-engineer-bootcamp, Student Handbook - Page 1, https://www.zaio.io/2027, Student Handbook - Page 7, https://www.zaio.io/intakes
11. **Where are ZAIO's campuses located?**: https://www.zaio.io/about, https://www.zaio.io/events, https://www.zaio.io/learner-stories, Student Handbook - Page 20
12. **Is ZAIO accredited?**: https://www.zaio.io/2027/parents, https://www.zaio.io/accreditation, https://www.zaio.io/, Student Handbook - Page 26, Student Handbook - Page 15
13. **What are the entry requirements for the occupational certificates?**: https://www.zaio.io/occupational-certificates, https://www.zaio.io/entry-requirements, https://www.zaio.io/2027, Student Handbook - Page 3
14. **How do I apply to ZAIO?**: https://www.zaio.io/how-to-apply, https://www.zaio.io/affiliate, https://www.zaio.io/terms, Student Handbook - Page 26, Student Handbook - Page 10
15. **What is The Weekly Commit?**: https://www.zaio.io/newsletter, Student Handbook - Page 7, https://www.zaio.io/intakes
16. **How much does the Fullstack AI Engineer bootcamp cost?**: https://www.zaio.io/programmes/fullstack-ai-engineer-bootcamp, Student Handbook - Page 1, https://www.zaio.io/programmes/ai-software-developer-occupational-certificate, https://www.zaio.io/programmes/cloud-devops-engineering-bootcamp, Student Handbook - Page 16
17. **Which companies can I use to finance my studies?**: Student Handbook - Page 17, Student Handbook - Page 16, https://www.zaio.io/programmes/fullstack-ai-engineer-bootcamp, https://www.zaio.io/programmes/fullstack-web-development-bootcamp, https://www.zaio.io/2027
18. **What is the capital of France?**: nothing above threshold
19. **Who won the 2022 FIFA World Cup?**: nothing above threshold
20. **What is the Wi-Fi password at the Cape Town campus?**: https://www.zaio.io/events/zaio-student-connect, https://www.zaio.io/locations/cape-town, Student Handbook - Page 7, Student Handbook - Page 26
21. **What will the weather be in Cape Town tomorrow?**: https://www.zaio.io/events/zaio-open-day-capetown, https://www.zaio.io/events/zaio-open-day-durban, https://www.zaio.io/locations/cape-town, Student Handbook - Page 7

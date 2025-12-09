from langchain_core.runnables import RunnableLambda

r = RunnableLambda(lambda x: x + 1)
result = r.invoke(1)
print(result)           # 2
s = RunnableLambda(lambda x: x*x)
result = s.batch([3,4,5])
print(result)

result = r.batch([1, 2, 3])
print(result)           # [2, 3, 4]



from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch

minus = RunnableLambda(lambda x: x-1)
inc = RunnableLambda(lambda x: x+1)
square = RunnableLambda(lambda x: x*x)
r = RunnableParallel(increased=inc, squared=square)
result = r.invoke(1)
print(result)

chain2 = RunnableSequence(inc, RunnableParallel(increased=inc, squared=square))
result = chain2.invoke(1)
print(result)


b = RunnableBranch(
    (lambda x: x<0, square),
    (lambda x: x>0, minus),
    inc
)

print(b.invoke(-2))
print(b.invoke(0))
print(b.invoke(11))


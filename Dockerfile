FROM openjdk:8-jre-slim

RUN pip3 install numpy && \
    pip3 install pandas && \
    pip3 install sklearn && \
    pip3 install pyspark && \
    pip3 install findspark

ENV PROG_DIR /winepredict
COPY test.py /winepredict/
COPY ValidationDataset.csv /winepredict/
COPY trainingmodel.model /winepredict/

ENV PROG_NAME test.py
ADD ${PROG_NAME} .

ENTRYPOINT ["python","test.py"]
CMD ["ValidationDataset.csv"]
# DRL-Based-DDoS-Attack-Detection-in-Cloud-Environment

Datasets used for Detection: 

CICDDoS2019 : https://www.kaggle.com/datasets/aymenabb/ddos-evaluation-dataset-cic-ddos2019

Description:

CICDDoS2019 is a dataset that contains benign and most up-to-date common DDoS attacks that resemble true real-world data (PCAPs). However, this version, in CSV file format, includes the results of network traffic analysis using CICFlowMeter-V3 with flows labeled based on timestamp, source and destination IPs, source and destination ports, protocols, and attack.

In this dataset, we have different modern reflective DDoS attacks such as PortMap, NetBIOS, LDAP, MSSQL, UDP, UDP-Lag, SYN, NTP, DNS and SNMP. Attacks were subsequently executed during this period. As Table III shows, we executed 12 DDoS attacks includes NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP, UDP-Lag, WebDDoS, SYN and TFTP on the training day and 7 attacks including PortScan, NetBIOS, LDAP, MSSQL, UDP, UDP-Lag and SYN in the testing day. The traffic volume for WebDDoS was so low and PortScan just has been executed in the testing day and will be unknown for evaluating the proposed model.



UNSW-NB15 : https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Description:

The raw network packets of the UNSW-NB 15 dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) for generating a hybrid of real modern normal activities and synthetic contemporary attack behaviours.

Tcpdump tool is utilised to capture 100 GB of the raw traffic (e.g., Pcap files). This dataset has nine types of attacks, namely, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms. The Argus, Bro-IDS tools are used and twelve algorithms are developed to generate totally 49 features with the class label.

These features are described in UNSW-NB15_features.csv file.

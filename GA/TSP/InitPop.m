function Chrom = InitPop( NIND,N )
%%��ʼ����Ⱥ
%���룺
%NIND����Ⱥ��С
%N������Ⱦɫ�峤�ȣ�����Ϊ���еĸ�����
%�����
%��ʼ��Ⱥ

Chrom=zeros(NIND,N);%���ڴ洢��Ⱥ
for i=1:NIND
    Chrom(i,:)=randperm(N);%������ɳ�ʼ��Ⱥ
end


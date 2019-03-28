//
//  main.cpp
//  A1107
//
//  Created by ergouu on 2019/3/27.
//  Copyright © 2019 ergouu. All rights reserved.
//

#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

vector<int> pl[1002],hby[1002],res;
bool visited[1002]={false};

bool cmp(const int &a,const int &b){
    return a>b;
}

void DFS(const int &v,int &count){
    if(!visited[v]){
        visited[v]=true;
        ++count;
        for(int c:hby[v]){//c表示第v个人的每一个hobby序号
            for(int d:pl[c])if(!visited[d])DFS(d,count);//对v来说，其所有拥有和其同一个序号的人都和他为一个cluster
        }
    }
}

void DFS_traverse(const int &n){
    for(int i=1;i<=n;++i){
        if(!visited[i]){
            int count=0;
            DFS(i,count);
            res.push_back(count);
        }
    }
}

int main(void){
    freopen("/Users/ergouu/Documents/code/in.log","r",stdin);
    int n;
    scanf("%d",&n);
    for(int i=1;i<=n;++i){
        int hnum;
        scanf("%d: ",&hnum);
        while(hnum--){
            int hid;
            scanf("%d",&hid);
            hby[i].push_back(hid);
            pl[hid].push_back(i);
        }
    }
    DFS_traverse(n);
    sort(res.begin(),res.end(),cmp);
    size_t size=res.size();
    cout<<size<<endl;
    for(int c:res)(--size)?printf("%d ",c):printf("%d",c);
    return 0;
}

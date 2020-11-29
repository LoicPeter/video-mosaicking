#ifndef UNIONFIND_INCLUDED
#define UNIONFIND_INCLUDED


class UnionFindObject
{
    public:
    void makeSet();
    UnionFindObject* findRepresentative();// finds (the address of) a representative of the current object
    void mergeWith(UnionFindObject *obj);

    private:
    UnionFindObject *m_parent;
    int m_rank;
};



#endif // UNIONFIND_INCLUDED

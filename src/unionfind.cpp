#include "unionfind.hpp"

void UnionFindObject::makeSet()
{
    this->m_parent = this;
    this->m_rank = 0;
}

UnionFindObject* UnionFindObject::findRepresentative()
{
    if (this->m_parent != this)
        this->m_parent = this->m_parent->findRepresentative();
    return this->m_parent;
}

void UnionFindObject::mergeWith(UnionFindObject *obj)
{
    UnionFindObject *current_representative = this->findRepresentative();
    UnionFindObject *obj_representative = obj->findRepresentative();

    // Only continue if they are not already in the same set
    if (current_representative != obj_representative)
    {
        if (current_representative->m_rank < obj_representative->m_rank)
            current_representative->m_parent = obj_representative;
        else
        {
            if (current_representative->m_rank > obj_representative->m_rank)
                obj_representative->m_parent = current_representative;
            else
            {
                obj_representative->m_parent = current_representative;
                current_representative->m_rank += 1;
            }
        }

    }

}
